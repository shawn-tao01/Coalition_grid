cd(joinpath(@__DIR__, ".."))   # the data loader uses paths relative to the repo root
include(joinpath(@__DIR__, "..", "Buildings.jl"))
include(joinpath(@__DIR__, "..", "MPC_optimiser.jl"))
include(joinpath(@__DIR__, "..", "Coalition.jl"))
include(joinpath(@__DIR__, "..", "load_EMS_data.jl"))
using Plots, Printf

# Candidates. Day 1, outage 05:00-12:00
#   [7, 19, 20]  
#   [19, 20, 13] 
#   [4, 7, 19]  
#   [7, 19, 22] 
#   [4, 5, 22]   
#   [19, 1, 4]   
#   [19, 25, 28] 
const IDS = [7, 19, 20]
const NAMES = ["building $(i)" for i in IDS]
const NUM_STEPS = 96     # day 1
const H = 8              # MPC horizon
const CMAX = 2           # max coalition size
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
One closed-loop day with fresh buildings (coal_MPC changes b.SoC in place). The coalition former is
wrapped to record the solver status of every plan it returns, so infeasible solves are counted, and the
soft-floor shortfall resil_short of every plan, summed over the coalitions of each step.
"""
function run_case(label, former; enabled, order = :paper, exact = false, fix = false)
    RESIL.enabled = enabled
    FORM.order, FORM.exact_bound, FORM.fix_dissolve = order, exact, fix
    bs, buy, sell = load_subset()
    global opt = MPC_optimiser(buy', sell')
    statuses = Any[]
    fails = Any[]   # every returned plan whose solve did not end OPTIMAL: step, coalition, solver statuses
    short_next = zeros(NUM_STEPS)   # [k+1]: resil_short[2] of the plans made at k, i.e. the applied state below s~C
    short_plan = zeros(NUM_STEPS)   # [k]: resil_short[2..H] summed, the whole shortfall planned at k
    recorded = function (bs, mcs, k, na, rh)
        coal, vars, iters = former(bs, mcs, k, na, rh) #former can be privacy_focused coals
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
    cost, _, _, _, hist, (shed, curt) = coal_MPC(recorded, bs, CMAX, H)
    return (label = label, bs = bs, cost = cost, hist = hist, time = time() - t0, resil = enabled,
            shed = sum(shed), curt = sum(curt),
            bad = count(s -> !(s in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)), statuses), solves = length(statuses),
            fails = fails, short_next = short_next, short_plan = short_plan)
end

"""
Step-by-step check of a run: at every step k, each coalition C chosen at k (a single building is a
coalition of one) must reach, at k+1, a total SoC of at least its floor s~_C from eq. (19). Returns
- the number of coalition-steps below s~_C;
- covered: member-steps in a pair where the member holds less than its own floor s~_{i} (what it would
  have to hold alone), i.e. its partner's battery covers part of its need;
- how often each pair formed;
- below: the coalition-steps under s~_C, with their gap.
Outage steps are skipped: there is no floor during an outage.
"""
function assess(r)
    bs = r.bs
    violations, covered = 0, 0
    pairs = Dict{String,Int}()
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
            length(m) > 1 || continue
            for i in m
                own = stilde_coal(bs[i:i], k, 1, RESIL.tau)[2]
                bs[i].SoC[k+1] < own - 1e-3 * Float64(bs[i].max_storage) && (covered += 1)
            end
            key = join(NAMES[sort(m)], " + ")
            pairs[key] = get(pairs, key, 0) + 1
        end
    end
    return violations, covered, pairs, below
end

results = [
    # run_case("decentralised, resilience off", decentralised; enabled = false),
    # run_case("as published, resilience off", privacy_focussed_coals; enabled = false),
    # run_case("decentralised, proactive", decentralised; enabled = true),
    # run_case("as published, proactive", privacy_focussed_coals; enabled = true),
    run_case("resilience first, proactive", resilience_focussed_coals; enabled = true),
    # run_case("Algorithm 1, proactive", privacy_focussed_coals; enabled = true, order = :val_desc, exact = true, fix = true),
]
FORM.order, FORM.exact_bound, FORM.fix_dissolve = :paper, false, false   # back to the defaults
checks = [assess(r) for r in results]

# ---------------------------------------------------------------- table
println()
@printf("%-32s %8s %6s | %8s %9s | %6s | %7s | %-11s | %s\n", "case", "cost", "time s", "shed", "curtailed",
        "below", "covered", "not OPTIMAL", "pairs formed (steps)")
ok = true
for (r, (viol, covered, pairs)) in zip(results, checks)
    r.resil && (viol > 0 || r.bad > 0) && (global ok = false)
    pair_str = isempty(pairs) ? "none" : join(["$p: $n" for (p, n) in sort(collect(pairs), by = x -> -x[2])], ", ")
    @printf("%-32s %8.2f %6.1f | %8.2f %9.2f | %6d | %7d | %4d of %4d | %s\n", r.label, r.cost, r.time,
            r.shed, r.curt, viol, covered, r.bad, r.solves, pair_str)
end
OUTAGE.enabled && println("outage at steps $(OUTAGE.start)-$(OUTAGE.stop): 'shed' and 'curtailed' are the realised totals over it")
println("\nas published = FORM defaults (smallest val first, vector min, last-agent dissolve flag);")
println("Algorithm 1 = largest val first, exact eq. (18), dissolve test (22) applied to every coalition.")
println("s~C = SEGAN eq. (19) without 1/N_ess for the coalition (or single building) at that step; 'below' counts")
println("coalition-steps whose total energy is under it (only enforced when proactive scheduling is on); 'covered'")
println("counts member-steps in a pair below the member's own floor, i.e. held for it by its partner;")
println("'not OPTIMAL' counts returned plans whose solve failed.")
println(ok ? "PASS: every proactive run solved every problem and kept every coalition at or above s~C" :
        "FAIL: a proactive run had a failed solve or a coalition below s~C")

# ---------------------------------------------------------------- which solves failed
"""
Re-solve the problem behind the failed plan `f` (step f.k, coalition f.members) with the constraint groups
in `soft` made soft: each constraint may be violated at a cost of `rho` per unit (JuMP's relax_with_penalty!).
The constraints that come out violated are the ones the hard problem could not satisfy together. coal_MPC
changed r.bs[i].SoC in place, but SoC[f.k] is still the state that solve started from, so the problem is the
same. Groups are the constraint names in single_optimise (:resil_floor, :charge_c, :power_c, ...); variable
bounds (0 <= charge <= capacity, flow limits) stay hard.
"""
function diagnose(r, f; soft = (:resil_floor,), rho = 100.0)
    RESIL.enabled = r.resil
    model, _ = single_optimise(opt, r.bs[f.members], f.k, H)
    pen = relax_with_penalty!(model, Dict(c => rho for g in soft for c in model[g]))
    optimize!(model)
    @printf("   re-solved with %s soft: %s\n", join(soft, ", "), termination_status(model))
    for (c, p) in sort(collect(pen), by = x -> name(x[1]))
        value(p) > 1e-4 && @printf("      %-18s violated by %.4f\n", name(c), value(p))   # value(p) is the violation itself
    end
end

for (r, ch) in zip(results, checks)
    r.resil && (r.bad > 0 || ch[1] > 0) || continue
    println("\n", r.label, ":")
    for f in r.fails
        @printf("   k = %2d  %-8s  solve ended %s (primal %s)\n", f.k, string(f.members), f.status, f.primal)
    end
    for b in ch[4]
        @printf("   k = %2d  %-8s  total energy at k+1 is %.4f below s~C\n", b.k, string(b.members), -b.gap)
    end
    isempty(r.fails) || diagnose(r, first(r.fails))   # others: diagnose(results[i], results[i].fails[j]; soft = (...))
end

# ---------------------------------------------------------------- plot
hours = (0:NUM_STEPS-1) ./ 4
on = [i for (i, r) in enumerate(results) if r.resil]
colours = (:gray, :black, :royalblue)
# start and end of the outage, as vertical red dashed lines (start of step OUTAGE.start, end of step OUTAGE.stop)
outage_lines!(p; label = "") = OUTAGE.enabled &&
    vline!(p, [(OUTAGE.start - 1) / 4, OUTAGE.stop / 4], color = :red, ls = :dash, lw = 1.5, label = label)
panels = Any[]
for i in eachindex(IDS)
    p = plot(title = NAMES[i], ylabel = "energy", legend = i == 1 ? :topright : false)
    for (j, col) in zip(on, colours)
        plot!(p, hours, results[j].bs[i].SoC, label = results[j].label, lw = 2, color = col, linetype = :steppost)
    end
    b = results[on[1]].bs[i]   # own floor (what the building must hold alone); none is set by an outage step
    own = [k == 1 || in_outage(k - 1) ? NaN : stilde_coal([b], k - 1, 1, RESIL.tau)[2] for k in 1:NUM_STEPS]
    plot!(p, hours, own, label = "own s~ (alone)", lw = 1.5, color = :darkorange, ls = :dash, linetype = :steppost)
    outage_lines!(p; label = "outage start / end")
    push!(panels, p)
end
psh = plot(ylabel = "reserve slack", legend = :topright)   # resil_short: total energy below s~C
for (j, col) in zip(on, colours)
    pre = length(on) > 1 ? results[j].label * ": " : ""
    plot!(psh, hours, results[j].short_next, label = pre * "slack, next step", lw = 2, color = col, linetype = :steppost)
    plot!(psh, hours, results[j].short_plan, label = pre * "slack, whole horizon", lw = 1, color = col, ls = :dash,
          linetype = :steppost)
end
outage_lines!(psh)
push!(panels, psh)
pair_code(coal) = begin
    ps = [sort(Int.(agent_ids(c))) for c in coal if length(agent_ids(c)) > 1]
    isempty(ps) ? 0 : Dict([1, 2] => 1, [1, 3] => 2, [2, 3] => 3)[ps[1]]
end
pp = plot(ylabel = "pair formed", xlabel = "hour of day", legend = :topright, ylims = (-0.4, 3.6),
          yticks = (0:3, ["none", "$(IDS[1]) + $(IDS[2])", "$(IDS[1]) + $(IDS[3])", "$(IDS[2]) + $(IDS[3])"]))
li_runs = [r for r in results if !occursin("decentralised", r.label)]
for (r, col, off) in zip(li_runs, (:orange, :black, :royalblue), (-0.08, 0.0, 0.08))
    plot!(pp, hours, [pair_code(r.hist[k]) + off for k in 1:NUM_STEPS], label = r.label, lw = 2, color = col, linetype = :steppost)
end
outage_lines!(pp)
fig = plot(panels..., pp, layout = grid(5, 1), size = (1000, 1600), left_margin = 8Plots.mm)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
savefig(fig, joinpath(outdir, "three_buildings.png"))
savefig(fig, joinpath(outdir, "three_buildings.pdf"))
println("saved ", joinpath(outdir, "three_buildings.png"), " and .pdf")
