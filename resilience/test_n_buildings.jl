cd(joinpath(@__DIR__, ".."))   # the data loader uses paths relative to the repo root
include(joinpath(@__DIR__, "..", "Buildings.jl"))
include(joinpath(@__DIR__, "..", "MPC_optimiser.jl"))
include(joinpath(@__DIR__, "..", "Coalition.jl"))
include(joinpath(@__DIR__, "..", "load_EMS_data.jl"))
using Plots, Printf

# Candidates
#   [4, 5, 7, 8, 9, 13, 14, 19, 20, 21, 22, 25]
#   [4, 5, 7, 8, 9, 13, 14, 19, 20, 21, 22, 28]
#   [4, 5, 7, 8, 9, 13, 14, 19, 20, 21, 22, 26]
#   [4, 5, 7, 8, 9, 13, 14, 15, 19, 20, 21, 22]
#   [4, 5, 7, 8, 9, 13, 14, 15, 19, 20, 21, 22]
#   [4, 5, 7, 8, 9, 13, 14, 15, 19, 20, 21, 22]
const IDS = [4, 5, 7, 8, 9, 13, 14, 19, 20, 21, 22, 25]
const NAMES = ["building $(i)" for i in IDS]
# const TYPES = ["solar farm", "office", "residential", "industrial", "warehouse pv", "data center", "school",
#                "restaurant", "coastal wind", "hilltop wind"][(IDS .- 1) .÷ 3 .+ 1]   # order of generate_synthetic_data.jl
const NUM_STEPS = 96     # day 1
const H = 8              # MPC horizon
const CMAX = 4           # max coalition size
const SOC0 = 0.5         # initial SoC as a share of capacity
const DATA_DIR = ""      # load_EMS_data.jl currently hardcodes synthetic_data/; use "synthetic_data" if you revert that
const OUT = "buildings_$(length(IDS))"   # output files: output/n_buildings/buildings_12.txt, ...
RESIL.tau = 4

OUTAGE.enabled = true
OUTAGE.start, OUTAGE.stop = 49, 72

"Load the chosen buildings and renumber them 1..N (coal_MPC indexes buy/sell by b.id)."
function load_subset()
    all_bs, buy, sell, _ = MPC_load_from_CSV(maximum(IDS), NUM_STEPS, DATA_DIR)
    bs = [MPC_Building(b.loc, b.pred_cons, b.pred_prod, b.act_cons, b.act_prod, b.max_storage, b.storage_max_flow,
                       b.charge_eff, b.discharge_eff, i, zeros(NUM_STEPS)) for (i, b) in enumerate(all_bs[IDS])]
    for b in bs
        b.SoC[1] = SOC0 * Float64(b.max_storage)
    end
    return bs, buy, sell
end

"Every building stays on its own (as in benchmark.jl); the outage problem during an outage."
function decentralised(bs, max_coal_size, k, num_look_ahead, receding_horizon = false)
    agents = Vector(1:length(bs))
    outs = [in_outage(k) ? (single_optimise_outage(opt, bs[a], k, num_look_ahead, receding_horizon)[2], 1) :
            single_optimise_ADMM(opt, bs[a], k, num_look_ahead, receding_horizon) for a in agents]
    return agents, [o[1] for o in outs], sum(o[2] for o in outs)
end

"""
Closed-loop simulation.
"""
function run_case(label, former; enabled, order = :paper, exact = false, fix = false)
    RESIL.enabled = enabled
    FORM.order, FORM.exact_bound, FORM.fix_dissolve = order, exact, fix
    bs, buy, sell = load_subset()
    global opt = MPC_optimiser(buy', sell')
    statuses = Any[]
    fails = Any[]   # every returned plan whose solve did not end OPTIMAL: step, coalition, solver statuses
    short_next = zeros(NUM_STEPS)   # [k+1]: soft var resil_short[2] of the plans made at k
    short_plan = zeros(NUM_STEPS)   # [k]:  soft var resil_short[2..H] summed
    step_time = zeros(NUM_STEPS)    # [k]: wall time of the coalition formation for Algorithm 1 at step k
    uc = fill(NaN, NUM_STEPS, length(IDS))   # [k, i]: energy charged into building i's battery at step k (NaN: no plan)
    ud = fill(NaN, NUM_STEPS, length(IDS))   # [k, i]: energy discharged from it
    recorded = function (bs, mcs, k, na, rh)
        t1 = time()
        coal, vars, iters = former(bs, mcs, k, na, rh) #former can be privacy_focused coals
        step_time[k] = time() - t1
        for (c, v) in zip(coal, vars)
            m = owner_model(first(v[4]))
            members = Int.(agent_ids(c))
            push!(statuses, termination_status(m))
            solved = last(statuses) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            solved || push!(fails, (k = k, members = members, status = last(statuses), primal = primal_status(m)))
            for (j, i) in enumerate(members)   # first step of the plan, the one applied
                uc[k, i] = solved ? value(m[:pos_delta_s][1, j]) : NaN
                ud[k, i] = solved ? -value(m[:neg_delta_s][1, j]) : NaN
            end
            if haskey(m, :resil_short) && !isempty(m[:resil_short]) && has_values(m)   # empty at the last step (H = 1)
                rs = value.(m[:resil_short])
                all(isfinite, rs) || continue
                short_next[k+1] += rs[2]
                short_plan[k] += sum(rs)
            end
        end
        return coal, vars, iters
    end

    t0 = time()
    cost, (buy, _), solves_per_step, _, hist, (shed, curt) = coal_MPC(recorded, bs, CMAX, H)

    return (label = label, bs = bs, cost = cost, hist = hist, time = time() - t0, resil = enabled,
            shed = sum(shed), curt = sum(curt), step_time = step_time, solves_per_step = solves_per_step,
            bad = count(s -> !(s in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)), statuses), solves = length(statuses),
            fails = fails, short_next = short_next, short_plan = short_plan, uc = uc, ud = ud, buy = buy)
end

"""
Step-by-step check of a run: at every step k, each coalition C must reach, at k+1, a total SoC of at least its floor s~_C from eq. (19). Returns
- the number of coalition-steps below s~_C;
- below: the coalition-steps under s~_C, with their gap.
"""
function assess(r)
    bs = r.bs
    violations = 0
    below = Any[]
    for k in 1:NUM_STEPS-1
        in_outage(k) && continue
        for c in r.hist[k]
            m = Int.(agent_ids(c))
            g = sum(bs[i].SoC[k+1] for i in m) - stilde_coal(bs[m], k, 1, RESIL.tau)[2]   # entry 2 = state at k+1
            if g < -1e-3 * sum(Float64(bs[i].max_storage) for i in m)
                violations += 1
                push!(below, (k = k, members = m, gap = g))
            end
        end
    end
    return violations, below
end

#Solve once before the timed runs so the timing do not include Julia's compilation
function warmup()
    RESIL.enabled = true
    bs, buy, sell = load_subset()
    global opt = MPC_optimiser(buy', sell')
    resilience_focussed_coals(bs, CMAX, 1, H)
    OUTAGE.enabled && resilience_focussed_coals(bs, CMAX, OUTAGE.start, H)
    return nothing
end
warmup()

results = [
    # run_case("decentralised, no reserve", decentralised; enabled = false),   # reference for cost and shedding
    run_case("decentralised, with reserve", decentralised; enabled = true),         # every building alone with its reserve
    run_case("resilience, with reserve", resilience_focussed_coals; enabled = true)
]
FORM.order, FORM.exact_bound, FORM.fix_dissolve = :paper, false, false   # back to the defaults
checks = [assess(r) for r in results]
alone = results[findfirst(r -> r.label == "decentralised, with reserve", results)]
main = results[findfirst(r -> startswith(r.label, "resilience"), results)]

# horizon length at step k, as in resilience_focussed_coals; clock time of step k
horizon(k) = max(min(length(pred_consumption(main.bs[1], k)), min(NUM_STEPS - k + 1, H)), 1)
clock(k) = @sprintf("%02d:%02d", (k - 1) ÷ 4, 15 * ((k - 1) % 4))
coalition_of(r, k, i) = first(a for a in r.hist[k] if i in Int.(agent_ids(a)))
# coalition run, outside the outage: steps where building i could not reach its own reserve alone (r < 0)
RESIL.enabled = true
needs = [[k for k in 1:NUM_STEPS if !in_outage(k) && reach_margin(main.bs[[i]], k, horizon(k)) < -RES_TOL] for i in eachindex(IDS)]

outdir = joinpath(@__DIR__, "output", "n_buildings")   # results of the runs with 12 or more buildings
mkpath(outdir)

# ---------------------------------------------------------------- table, written to output/n_buildings/buildings_N.txt
table_file = joinpath(outdir, "$(OUT).txt")
ok = true
open(table_file, "w") do io
    @printf(io, "%-32s %8s | %8s %9s | %10s %9s %6s | %s\n", "case", "cost", "shed", "curtailed",
            "t_mean (s)", "t_max (s)", "solves", "not OPTIMAL")
    for (r, (viol, _)) in zip(results, checks)
        startswith(r.label, "resilience") && (viol > 0 || r.bad > 0) && (global ok = false)
        @printf(io, "%-32s %8.2f | %8.2f %9.2f | %10.3f %9.3f %6.1f | %4d of %4d\n", r.label, r.cost,
                r.shed, r.curt, sum(r.step_time) / NUM_STEPS, maximum(r.step_time), r.solves_per_step,
                r.bad, r.solves)
    end
    println(io)
    for (col, text) in [
            "cost"        => "realised grid cost",
            "shed"        => "shedded load summed over buildings and steps",
            "curtailed"   => "generation curtailed, summed over buildings and steps",
            "t_mean (s)"  => "wall time of the coalition formation, the mean over the $(NUM_STEPS) steps",
            "t_max (s)"   => "the slowest of the $(NUM_STEPS) steps",
            "solves"      => "optimisation problems solved at one step, mean over the $(NUM_STEPS) steps",
            "not OPTIMAL" => "steps when solver status is neither OPTIMAL nor ALMOST_OPTIMAL"]
        @printf(io, "%-13s%s\n", col * ":", text)
    end
    OUTAGE.enabled && println(io, "outage at steps $(OUTAGE.start)-$(OUTAGE.stop)")
    println(io, "Cmax $(CMAX), buildings ", join(IDS, ", "))
    println(io, ok ? "PASS" : "FAIL: the coalition run had a failed solve or a coalition below s~C")
    for i in eachindex(IDS)
        ks = [f.k for f in alone.fails if i in f.members]
        isempty(ks) && continue
        covered = count(k -> length(agent_ids(coalition_of(main, k, i))) > 1 &&
                             !any(f -> f.k == k && i in f.members, main.fails), needs[i])
        @printf(io, "  %-12s failed alone at %2d steps, first %s | coalition run: failed %d, not reachable alone %2d, covered %2d\n",
                NAMES[i], length(ks), clock(first(ks)), count(f -> i in f.members, main.fails),
                length(needs[i]), covered)
    end
end
println("saved ", table_file)

# """
# Re-solve the problem behind the failed plan `f` of run `r`.
# """
# function diagnose(r, f; soft = (:resil_floor,), rho = 100.0)
#     RESIL.enabled = r.resil
#     model, _ = single_optimise(opt, r.bs[f.members], f.k, H)
#     pen = relax_with_penalty!(model, Dict(c => rho for g in soft for c in model[g]))
#     optimize!(model)
#     @printf("   re-solved with %s soft: %s\n", join(soft, ", "), termination_status(model))
#     for (c, p) in sort(collect(pen), by = x -> name(x[1]))
#         value(p) > 1e-4 && @printf("      %-18s violated by %.4f\n", name(c), value(p))   # value(p) is the violation itself
#     end
# end

# for (r, ch) in zip(results, checks)
#     startswith(r.label, "resilience") && (r.bad > 0 || ch[1] > 0) || continue
#     println("\n", r.label, ":")
#     for f in r.fails
#         @printf("   k = %2d  %-8s  solve ended %s (primal %s)\n", f.k, string(f.members), f.status, f.primal)
#     end
#     for b in ch[2]
#         @printf("   k = %2d  %-8s  total energy at k+1 is %.4f below s~C\n", b.k, string(b.members), -b.gap)
#     end
#     isempty(r.fails) || diagnose(r, first(r.fails))   # others: diagnose(results[i], results[i].fails[j]; soft = (...))
# end

# ---------------------------------------------------------------- plot
hours = (0:NUM_STEPS-1) ./ 4
on = [i for (i, r) in enumerate(results) if r.resil]
colours = (:gray, :black, :royalblue)
# start and end of the outage, as vertical red dashed lines (start of step OUTAGE.start, end of step OUTAGE.stop)
outage_lines!(p; label = "") = OUTAGE.enabled &&
    vline!(p, [(OUTAGE.start - 1) / 4, OUTAGE.stop / 4], color = :red, ls = :dash, lw = 1.5, label = label)
# stored energy of building i in run r; a decentralised run is drawn only up to its first step without a plan
function curve(r, i)
    soc = copy(r.bs[i].SoC)
    ks = [f.k for f in r.fails if i in f.members]
    startswith(r.label, "decentralised") && !isempty(ks) && (soc[first(ks)+1:end] .= NaN)
    return soc
end

# figures of the coalition run, the coalitions at every step and the load plots, in output/n_buildings/combine
combinedir = joinpath(outdir, "combine")
mkpath(combinedir)

# three buildings per figure: stored energy alone (gray) and in the coalition run (black)
for (f, group) in enumerate(Iterators.partition(eachindex(IDS), 3))
    panels = Any[]
    for i in group
        p = plot(title = NAMES[i], ylabel = "energy", xlabel = i == last(group) ? "hour of day" : "",
                 legend = i == first(group) ? :topright : false)
        for (j, col) in zip(on, colours)
            plot!(p, hours, curve(results[j], i), label = results[j].label, lw = 2, color = col, linetype = :steppost)
        end
        b = results[on[1]].bs[i]   # own floor (what the building must hold alone); none is set by an outage step
        own = [k == 1 || in_outage(k - 1) ? NaN : stilde_coal([b], k - 1, 1, RESIL.tau)[2] for k in 1:NUM_STEPS]
        plot!(p, hours, own, label = "own tilde_s", lw = 1.5, color = :darkorange, ls = :dash, linetype = :steppost)
        outage_lines!(p; label = "outage start / end")
        push!(panels, p)
    end
    fig_b = plot(panels..., layout = grid(length(group), 1), size = (1050, 320 * length(group)), left_margin = 8Plots.mm)
    savefig(fig_b, joinpath(combinedir, "$(OUT)_$(f).png"))
    savefig(fig_b, joinpath(combinedir, "$(OUT)_$(f).pdf"))
    println("saved ", joinpath(combinedir, "$(OUT)_$(f).png"), " and .pdf")
end

# plot each building infeasible when alone
singledir = joinpath(outdir, "single")
mkpath(singledir)
upto(n) = [hours[1:n]; n / 4]   # starts of steps 1..n and the end of step n, so that :steppost draws step n too
headroom(v) = (lo = minimum(filter(isfinite, v)); hi = maximum(filter(isfinite, v)); (lo - 0.05 * (hi - lo), hi + 0.6 * (hi - lo) + (hi == lo)))
for i in eachindex(IDS)
    ks = [f.k for f in alone.fails if i in f.members]
    isempty(ks) && continue
    kf, b = first(ks), alone.bs[i]   # kf: first step whose plan alone has no solution
    xr = (0, kf / 4)
    p1 = plot(title = NAMES[i] * " alone", ylabel = "energy", legend = :topleft, xlims = xr)
    p2 = plot(ylabel = "energy", legend = :topleft, xlims = xr)
    p3 = plot(ylabel = "energy", xlabel = "hour of day", legend = :topleft, xlims = xr)
    if kf > 1   # the steps whose plan was applied; no plan from the first step: a blank figure
        n = kf - 1
        plot!(p1, upto(kf), [b.SoC[1:kf]; b.SoC[kf]], label = "stored energy", lw = 2, color = :gray, linetype = :steppost)
        own = [k == 1 || in_outage(k - 1) ? NaN : stilde_coal([b], k - 1, 1, RESIL.tau)[2] for k in 1:kf+1]
        plot!(p1, upto(kf), own, label = "own tilde_s", lw = 1.5, color = :darkorange, ls = :dash, linetype = :steppost)
        plot!(p2, upto(n), [alone.uc[1:n, i]; alone.uc[n, i]], label = "charge", lw = 2, color = :seagreen, linetype = :steppost)
        plot!(p2, upto(n), [alone.ud[1:n, i]; alone.ud[n, i]], label = "discharge", lw = 2, color = :firebrick, linetype = :steppost)
        plot!(p3, upto(n), [alone.buy[1:n, i]; alone.buy[n, i]], label = "bought from the grid", lw = 2, color = :royalblue,
              linetype = :steppost)
        plot!(p1, ylims = headroom([b.SoC[1:kf]; own]))   # room above the data for the legend
        plot!(p2, ylims = headroom([0; alone.uc[1:n, i]; alone.ud[1:n, i]]))
        plot!(p3, ylims = headroom([0; alone.buy[1:n, i]]))
        for (p, lab) in ((p1, "outage start / end"), (p2, ""), (p3, ""))
            OUTAGE.enabled && (OUTAGE.start - 1) / 4 < kf / 4 && outage_lines!(p; label = lab)
        end
    else
        foreach(p -> plot!(p, ylims = (0, 1)), (p1, p2, p3))
    end
    fig_s = plot(p1, p2, p3, layout = grid(3, 1), size = (1050, 900), left_margin = 8Plots.mm)
    savefig(fig_s, joinpath(singledir, "$(OUT)_building_$(IDS[i]).png"))
    savefig(fig_s, joinpath(singledir, "$(OUT)_building_$(IDS[i]).pdf"))
    println("saved ", joinpath(singledir, "$(OUT)_building_$(IDS[i]).png"), " and .pdf")
end

# coalitions of the coalition run at every step, in output/n_buildings/combine
coal_file = joinpath(combinedir, "$(OUT)_coalitions.txt")
open(coal_file, "w") do io
    println(io, "coalitions of the run \"$(main.label)\", Cmax $(CMAX); building ids, a + joins the members of a coalition")
    for k in 1:NUM_STEPS
        groups = sort([sort(IDS[Int.(agent_ids(a))]) for a in main.hist[k]], by = first)
        @printf(io, "%2d  %s%s  %s\n", k, clock(k), in_outage(k) ? "  outage" : "        ", join(join.(groups, "+"), "  "))
    end
end
println("saved ", coal_file)

# load and generation of the buildings
let script = joinpath(@__DIR__, "plot_buildings.py"),
    python = something(Sys.which("python"), Sys.which("py"), "python"),
    args = ["-b", string.(IDS)..., "--outdir", combinedir, "--format", "png", "pdf"]
    OUTAGE.enabled && append!(args, ["--outage", string(OUTAGE.start), string(OUTAGE.stop)])
    try
        run(`$python $script $args`)
    catch e
        println("could not run plot_buildings.py: ", e)
    end
end
