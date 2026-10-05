cd(joinpath(@__DIR__, ".."))   # the data loader uses paths relative to the repo root
include(joinpath(@__DIR__, "..", "Buildings.jl"))
include(joinpath(@__DIR__, "..", "MPC_optimiser.jl"))
include(joinpath(@__DIR__, "..", "load_EMS_data.jl"))
using Plots, Printf

const BUILDING = 19      # synthetic school
const NUM_STEPS = 96     # day 1
const H = 8              # MPC horizon, as in the paper
const SOC0 = 0.5         # initial SoC as a share of capacity
const DATA_DIR = ""      # load_EMS_data.jl currently hardcodes synthetic_data/; use "synthetic_data" if you revert that
RESIL.tau = 4


"""
Run the receding-horizon loop with resilience on or off. `target[k]` is the s~ that the solve at
step k-1 scheduled for step k (the value the floor charge[2] >= s~[2] enforces) `cost` is the realised grid cost, computed as in optimise().
"""
function run_closed_loop(enabled::Bool)
    RESIL.enabled = enabled
    bs, buy, sell, _ = MPC_load_from_CSV(BUILDING, NUM_STEPS, DATA_DIR)
    b = bs[BUILDING]
    global opt = MPC_optimiser(buy', sell')
    b.SoC[1] = SOC0 * Float64(b.max_storage)
    target = fill(NaN, NUM_STEPS)
    status = Any[]
    objective_val = 0.0
    energy_cost = 0.0

    for k in 1:NUM_STEPS-1
        Hk = min(H, NUM_STEPS - k + 1)   # single_optimise shortens the horizon at the end of the data
        target[k+1] = stilde_coal([b], k, Hk, RESIL.tau)[2]   # the call single_optimise makes (SEGAN eq. 19)
        model, res = single_optimise(opt, [b], k, H)
        push!(status, termination_status(model))
        b.SoC[k+1] = max(0, value(res[4][2, 1]))
        # remaining = b.act_prod[k] - b.act_cons[k] - (value(res[6][1, 1]) + value(res[1][1, 1]))
        energy_cost += buy[k] * max(0, value(res[2][1, 1])) - sell[k] * max(0, value(res[3][1, 1]))
        # cost += remaining > 0 ? -sell[k] * remaining : -buy[k] * remaining
    end
    return b, target, status, energy_cost
end

b, target, status, energy_cost_on = run_closed_loop(true)
b_off, _, _, energy_cost_off = run_closed_loop(false)
RESIL.enabled = true

# ---------------------------------------------------------------- checks
Smax = Float64(b.max_storage)
tol = 1e-3 * Smax                                  # SCS solves to about 1e-4
ks = 2:NUM_STEPS
gap = b.SoC[ks] .- target[ks]
solved = all(s -> s in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL), status)
held = minimum(gap) >= -tol
@printf("building %d, %d steps, horizon %d, tau %d\n", BUILDING, NUM_STEPS, H, RESIL.tau)
@printf("solver status: %s\n", join(unique(string.(status)), ", "))
@printf("SoC - s~: min %.4f, mean %.3f; floor active (within tol) at %d of %d steps\n", minimum(gap), sum(gap) / length(gap),
        count(abs.(gap) .<= tol), length(gap))
@printf("without resilience the SoC is below this s~ at %d of %d steps\n", count(b_off.SoC[ks] .< target[ks] .- tol), length(ks))
@printf("total energy cost: %.2f (resilience on), %.2f (resilience off)\n", energy_cost_on, energy_cost_off)
println(solved && held ? "PASS: single_optimise keeps the SoC at or above s~ at every step" :
        "FAIL: " * (solved ? "SoC fell below s~" : "a solve did not reach OPTIMAL"))

# ---------------------------------------------------------------- plot
# SoC and s~ are both values at the start of each step, so draw both as steps held over that step
# (:steppost). Mixing straight lines for SoC with a :steppre staircase for s~ made the SoC look
# like it dipped below s~ between samples, although it never does at any step.
hours = (0:NUM_STEPS-1) ./ 4
p1 = plot(hours, b.SoC, label = "SoC, resilience on", lw = 2.5, color = :royalblue, linetype = :steppost,
          ylabel = "state of charge", legend = :outertopright,
          title = "single_optimise, building $BUILDING (synthetic school), day 1")
plot!(p1, hours, target, label = "scheduled s~", lw = 2, color = :darkorange, linetype = :steppost)
plot!(p1, hours, b_off.SoC, label = "SoC, resilience off", lw = 1.5, color = :gray, ls = :dash, linetype = :steppost)
active = [k for k in ks if abs(b.SoC[k] - target[k]) <= tol]
scatter!(p1, hours[active], b.SoC[active], label = "floor active (SoC = s~)", color = :black, ms = 4)
hline!(p1, [Smax], label = "capacity", color = :black, lw = 1, ls = :dashdot)
p2 = plot(hours, b.act_cons, label = "load (actual)", lw = 2, color = :firebrick, legend = :outertopright,
          xlabel = "hour of day", ylabel = "energy per step")
plot!(p2, hours, b.act_prod, label = "PV (actual)", lw = 2, color = :seagreen)
fig = plot(p1, p2, layout = grid(2, 1, heights = [0.62, 0.38]), size = (1000, 700), left_margin = 5Plots.mm)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
savefig(fig, joinpath(outdir, "single_optimise_soc_b$(BUILDING).png"))
savefig(fig, joinpath(outdir, "single_optimise_soc_b$(BUILDING).pdf"))
println("saved ", joinpath(outdir, "single_optimise_soc_b$(BUILDING).png"), " and .pdf")
