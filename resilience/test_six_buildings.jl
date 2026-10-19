cd(joinpath(@__DIR__, ".."))   # the data loader uses paths relative to the repo root
include(joinpath(@__DIR__, "..", "Buildings.jl"))
include(joinpath(@__DIR__, "..", "MPC_optimiser.jl"))
include(joinpath(@__DIR__, "..", "Coalition.jl"))
include(joinpath(@__DIR__, "..", "load_EMS_data.jl"))
using Plots, Printf

# Candidates
#   [7, 19, 20, 4, 5, 22]
#   [7, 8, 9, 19, 20, 21]
#   [4, 5, 7, 8, 19, 20]
#   [19, 4, 2, 3, 5, 6]
#   [4, 7, 13, 19, 22, 23]
#   [4, 7, 13, 19, 22, 20]
#   [7, 19, 22, 4, 5, 13]
#   [4, 7, 13, 19, 22, 5]
#   [4, 7, 13, 19, 22, 8]
#   [4, 7, 13, 19, 22, 14]
#   [7, 8, 19, 20, 13, 14]
#   [7, 19, 20, 13, 22, 23]
#   [7, 8, 19, 20, 22, 23]
#   [4, 5, 6, 7, 8, 9]
#   [19, 20, 21, 22, 23, 24]
#   [7, 19, 20, 1, 4, 22]
const IDS = [7, 19, 20, 4, 5, 22]
const NAMES = ["building $(i)" for i in IDS]
const NUM_STEPS = 96     # day 1
const H = 8              # MPC horizon
const CMAX = 5           # max coalition size
const SOC0 = 0.5         # initial SoC as a share of capacity
const DATA_DIR = ""      # load_EMS_data.jl currently hardcodes synthetic_data/; use "synthetic_data" if you revert that
RESIL.tau = 4

OUTAGE.enabled = true
OUTAGE.start, OUTAGE.stop = 21, 48

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
    recorded = function (bs, mcs, k, na, rh)
        t1 = time()
        coal, vars, iters = former(bs, mcs, k, na, rh) #former can be privacy_focused coals
        step_time[k] = time() - t1
        for (c, v) in zip(coal, vars)
            m = owner_model(first(v[4]))
            push!(statuses, termination_status(m))
            last(statuses) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ||
                push!(fails, (k = k, members = Int.(agent_ids(c)), status = last(statuses), primal = primal_status(m)))
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
    cost, _, solves_per_step, _, hist, (shed, curt) = coal_MPC(recorded, bs, CMAX, H)

    return (label = label, bs = bs, cost = cost, hist = hist, time = time() - t0, resil = enabled,
            shed = sum(shed), curt = sum(curt), step_time = step_time, solves_per_step = solves_per_step,
            bad = count(s -> !(s in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)), statuses), solves = length(statuses),
            fails = fails, short_next = short_next, short_plan = short_plan)
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
    run_case("decentralised, no reserve", decentralised; enabled = false),   # reference for cost and shedding
    # run_case("as published, no reserve", privacy_focussed_coals; enabled = false),
    run_case("resilience, with reserve", resilience_focussed_coals; enabled = true)
]
FORM.order, FORM.exact_bound, FORM.fix_dissolve = :paper, false, false   # back to the defaults
checks = [assess(r) for r in results]

# ---------------------------------------------------------------- table, written to output/six_buildings.txt
table_file = joinpath(@__DIR__, "output", "six_buildings.txt")
mkpath(dirname(table_file))
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
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)

# three buildings per figure: stored energy in the coalition run
for (f, group) in enumerate(Iterators.partition(eachindex(IDS), 3))
    panels = Any[]
    for i in group
        p = plot(title = NAMES[i], ylabel = "energy", xlabel = i == last(group) ? "hour of day" : "",
                 legend = i == first(group) ? :topright : false)
        for (j, col) in zip(on, colours)
            plot!(p, hours, curve(results[j], i), label = "proactive scheduling", lw = 2, color = col, linetype = :steppost)
        end
        b = results[on[1]].bs[i]   # own floor (what the building must hold alone); none is set by an outage step
        own = [k == 1 || in_outage(k - 1) ? NaN : stilde_coal([b], k - 1, 1, RESIL.tau)[2] for k in 1:NUM_STEPS]
        plot!(p, hours, own, label = "own tilde_s", lw = 1.5, color = :darkorange, ls = :dash, linetype = :steppost)
        outage_lines!(p; label = "outage start / end")
        push!(panels, p)
    end
    fig_b = plot(panels..., layout = grid(length(group), 1), size = (1050, 320 * length(group)), left_margin = 8Plots.mm)
    savefig(fig_b, joinpath(outdir, "six_buildings_$(f).png"))
    savefig(fig_b, joinpath(outdir, "six_buildings_$(f).pdf"))
    println("saved ", joinpath(outdir, "six_buildings_$(f).png"), " and .pdf")
end

# figure 3: soft var and the pairs formed
psh = plot(ylabel = "reserve slack", legend = :topright)   # resil_short: total energy below s~C
for (j, col) in zip(on, colours)
    pre = length(on) > 1 ? results[j].label * ": " : ""
    plot!(psh, hours, results[j].short_next, label = pre * "slack var of the s>=tilde_s, on next step only", lw = 2, color = col, linetype = :steppost)
end
outage_lines!(psh)
li_runs = [r for r in results if !occursin("decentralised", r.label)]
coal_key(a) = sort(Int.(agent_ids(a)))
# one row per coalition formed at some step; row 0: steps where every building is alone
formed = sort(unique(coal_key(a) for r in li_runs for c in r.hist for a in c if length(agent_ids(a)) > 1), by = c -> sort(IDS[c]))
pp = plot(ylabel = CMAX == 2 ? "pair formed" : "coalition formed", xlabel = "hour of day", legend = :topright,
          ylims = (-0.6, length(formed) + 0.6), yticks = (0:length(formed), ["none"; [join(sort(IDS[c]), " + ") for c in formed]]))
for (r, col, off) in zip(li_runs, (:orange, :black, :royalblue), (-0.08, 0.0, 0.08))
    none = [all(a -> length(agent_ids(a)) == 1, r.hist[k]) ? off : NaN for k in 1:NUM_STEPS]
    plot!(pp, hours, none, label = "proactive scheduling", lw = 4, color = col, linetype = :steppost)
    for (row, c) in enumerate(formed)
        plot!(pp, hours, [any(a -> coal_key(a) == c, r.hist[k]) ? row + off : NaN for k in 1:NUM_STEPS],
              label = "", lw = 4, color = col, linetype = :steppost)
    end
end
outage_lines!(pp)
fig = plot(psh, pp, layout = grid(2, 1, heights = [0.3, 0.7]), size = (1050, 960), left_margin = 8Plots.mm)
savefig(fig, joinpath(outdir, "six_buildings_pairs.png"))
savefig(fig, joinpath(outdir, "six_buildings_pairs.pdf"))
println("saved ", joinpath(outdir, "six_buildings_pairs.png"), " and .pdf")

# load and generation of the buildings
let script = joinpath(@__DIR__, "plot_buildings.py"),
    python = something(Sys.which("python"), Sys.which("py"), "python"),
    args = ["-b", string.(IDS)..., "--outdir", outdir, "--format", "png", "pdf"]
    OUTAGE.enabled && append!(args, ["--outage", string(OUTAGE.start), string(OUTAGE.stop)])
    try
        run(`$python $script $args`)
    catch e
        println("could not run plot_buildings.py: ", e)
    end
end
