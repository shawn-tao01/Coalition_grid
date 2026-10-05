cd(joinpath(@__DIR__, ".."))   # the data loader uses paths relative to the repo root
include(joinpath(@__DIR__, "..", "Buildings.jl"))
include(joinpath(@__DIR__, "..", "MPC_optimiser.jl"))
include(joinpath(@__DIR__, "..", "Coalition.jl"))
include(joinpath(@__DIR__, "..", "load_EMS_data.jl"))
using Plots, Printf

const IDS = [19, 1, 4]
const NAMES = ["school 19", "solar farm 1", "office 4"]
const NUM_STEPS = 96     # day 1
const H = 8              # MPC horizon
const CMAX = 2           # max coalition size
const SOC0 = 0.5         # initial SoC as a share of capacity
const DATA_DIR = ""      # load_EMS_data.jl currently hardcodes synthetic_data/; use "synthetic_data" if you revert that
RESIL.tau = 4

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

"Every building stays on its own (as in benchmark.jl)."
function decentralised(bs, max_coal_size, k, num_look_ahead, receding_horizon = false)
    agents = Vector(1:length(bs))
    outs = [single_optimise_ADMM(opt, bs[a], k, num_look_ahead, receding_horizon) for a in agents]
    return agents, [o[1] for o in outs], sum(o[2] for o in outs)
end

"""
One closed-loop day with fresh buildings (coal_MPC changes b.SoC in place). The coalition former is
wrapped to record the solver status of every plan it returns, so infeasible solves are counted.
"""
function run_case(label, former; enabled, order = :paper, exact = false, fix = false)
    RESIL.enabled = enabled
    FORM.order, FORM.exact_bound, FORM.fix_dissolve = order, exact, fix
    bs, buy, sell = load_subset()
    global opt = MPC_optimiser(buy', sell')
    statuses = Any[]
    recorded = function (bs, mcs, k, na, rh)
        coal, vars, iters = former(bs, mcs, k, na, rh)
        append!(statuses, [termination_status(owner_model(first(v[4]))) for v in vars])
        return coal, vars, iters
    end
    t0 = time()
    cost, _, _, _, hist = coal_MPC(recorded, bs, CMAX, H)
    return (label = label, bs = bs, cost = cost, hist = hist, time = time() - t0, resil = enabled,
            bad = count(s -> !(s in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)), statuses), solves = length(statuses))
end

"""
Step-by-step check of a run: at every step k, each coalition C chosen at k (a single building is a
coalition of one) must reach, at k+1, a total SoC of at least its floor s~_C from eq. (19). Returns
- margin[k+1]: the smallest total SoC - s~_C over the coalitions of step k (NaN at k = 1);
- the number of coalition-steps below s~_C;
- covered: member-steps in a pair where the member holds less than its own floor s~_{i} (what it would
  have to hold alone), i.e. its partner's battery covers part of its need;
- how often each pair formed.
"""
function assess(r)
    bs = r.bs
    margin = fill(NaN, NUM_STEPS)
    violations, covered = 0, 0
    pairs = Dict{String,Int}()
    for k in 1:NUM_STEPS-1
        for c in r.hist[k]
            m = Int.(agent_ids(c))
            g = sum(bs[i].SoC[k+1] for i in m) - stilde_coal(bs[m], k, 1, RESIL.tau)[2]   # entry 2 = state at k+1
            margin[k+1] = isnan(margin[k+1]) ? g : min(margin[k+1], g)
            g < -1e-3 * sum(Float64(bs[i].max_storage) for i in m) && (violations += 1)
            length(m) > 1 || continue
            for i in m
                own = stilde_coal(bs[i:i], k, 1, RESIL.tau)[2]
                bs[i].SoC[k+1] < own - 1e-3 * Float64(bs[i].max_storage) && (covered += 1)
            end
            key = join(NAMES[sort(m)], " + ")
            pairs[key] = get(pairs, key, 0) + 1
        end
    end
    return margin, violations, covered, pairs
end

results = [
    run_case("decentralised, resilience off", decentralised; enabled = false),
    run_case("as published, resilience off", privacy_focussed_coals; enabled = false),
    run_case("decentralised, proactive", decentralised; enabled = true),
    run_case("as published, proactive", privacy_focussed_coals; enabled = true),
    run_case("Algorithm 1, proactive", privacy_focussed_coals; enabled = true, order = :val_desc, exact = true, fix = true),
]
FORM.order, FORM.exact_bound, FORM.fix_dissolve = :paper, false, false   # back to the defaults
checks = [assess(r) for r in results]

# ---------------------------------------------------------------- table
println()
@printf("%-32s %8s %6s | %12s | %6s | %7s | %-11s | %s\n", "case", "cost", "time s", "min ΣSoC-s~C",
        "below", "covered", "not OPTIMAL", "pairs formed (steps)")
ok = true
for (r, (margin, viol, covered, pairs)) in zip(results, checks)
    r.resil && (viol > 0 || r.bad > 0) && (global ok = false)
    pair_str = isempty(pairs) ? "none" : join(["$p: $n" for (p, n) in sort(collect(pairs), by = x -> -x[2])], ", ")
    @printf("%-32s %8.2f %6.1f | %12.2f | %6d | %7d | %4d of %4d | %s\n", r.label, r.cost, r.time,
            minimum(filter(!isnan, margin)), viol, covered, r.bad, r.solves, pair_str)
end
println("\nas published = FORM defaults (smallest val first, vector min, last-agent dissolve flag);")
println("Algorithm 1 = largest val first, exact eq. (18), dissolve test (22) applied to every coalition.")
println("s~C = SEGAN eq. (19) without 1/N_ess for the coalition (or single building) at that step; 'below' counts")
println("coalition-steps whose total SoC is under it (only enforced when proactive scheduling is on); 'covered'")
println("counts member-steps in a pair below the member's own floor, i.e. held for it by its partner;")
println("'not OPTIMAL' counts returned plans whose solve failed.")
println(ok ? "PASS: every proactive run solved every problem and kept every coalition at or above s~C" :
        "FAIL: a proactive run had a failed solve or a coalition below s~C")

# ---------------------------------------------------------------- plot
hours = (0:NUM_STEPS-1) ./ 4
on = [i for (i, r) in enumerate(results) if r.resil]
colours = (:gray, :black, :royalblue)
panels = Any[]
for i in eachindex(IDS)
    p = plot(title = NAMES[i], ylabel = "state of charge", legend = i == 1 ? :topright : false)
    for (j, col) in zip(on, colours)
        plot!(p, hours, results[j].bs[i].SoC, label = results[j].label, lw = 2, color = col, linetype = :steppost)
    end
    b = results[on[1]].bs[i]   # own floor (what the building must hold alone)
    own = [k == 1 ? NaN : stilde_coal([b], k - 1, 1, RESIL.tau)[2] for k in 1:NUM_STEPS]
    plot!(p, hours, own, label = "own s~ (alone)", lw = 1.5, color = :darkorange, ls = :dash, linetype = :steppost)
    push!(panels, p)
end
pm = plot(title = "tightest coalition reserve: min over coalitions of (total SoC - s~C)", ylabel = "energy",
          legend = :topright)
for (j, col) in zip(on, colours)
    plot!(pm, hours, checks[j][1], label = results[j].label, lw = 2, color = col, linetype = :steppost)
end
hline!(pm, [0.0], label = "", color = :darkorange, ls = :dash)
push!(panels, pm)
pair_code(coal) = begin
    ps = [sort(Int.(agent_ids(c))) for c in coal if length(agent_ids(c)) > 1]
    isempty(ps) ? 0 : Dict([1, 2] => 1, [1, 3] => 2, [2, 3] => 3)[ps[1]]
end
pp = plot(title = "pair formed at each step", xlabel = "hour of day", legend = :topright,
          yticks = (0:3, ["none", "school + solar", "school + office", "solar + office"]), ylims = (-0.4, 3.6))
li_runs = [r for r in results if !occursin("decentralised", r.label)]
for (r, col, off) in zip(li_runs, (:orange, :black, :royalblue), (-0.08, 0.0, 0.08))
    plot!(pp, hours, [pair_code(r.hist[k]) + off for k in 1:NUM_STEPS], label = r.label, lw = 2, color = col, linetype = :steppost)
end
fig = plot(panels..., pp, layout = grid(5, 1), size = (1000, 1600), left_margin = 8Plots.mm)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
savefig(fig, joinpath(outdir, "three_buildings.png"))
savefig(fig, joinpath(outdir, "three_buildings.pdf"))
println("saved ", joinpath(outdir, "three_buildings.png"), " and .pdf")
