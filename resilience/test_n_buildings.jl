# Section IV of eccpaper.tex: julia --project=. resilience/test_n_buildings.jl [--figures: redraw from runs.jls] [--evening: outage 15:30-20:45]
cd(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "Buildings.jl"))
include(joinpath(@__DIR__, "..", "MPC_optimiser.jl"))
include(joinpath(@__DIR__, "..", "Coalition.jl"))
include(joinpath(@__DIR__, "..", "load_EMS_data.jl"))
using Plots, Printf, Statistics, Serialization

const IDS = [4, 5, 7, 8, 9, 13, 14, 19, 20, 21, 22, 25]
const NAMES = ["building $(i)" for i in IDS]
const LABELS = let arch = (IDS .- 1) .÷ 3 .+ 1, types = unique(arch)
    [string('a' + findfirst(==(t), types) - 1, count(==(t), arch[1:i])) for (i, t) in enumerate(arch)]
end
const NUM_STEPS = 96
const H = 8
const CMAX = 4
const CMAX_SWEEP = [2, 3, 5]
const SOC0 = 0.5
const DATA_DIR = ""
const OUT = "buildings_$(length(IDS))"
const EVENING = "--evening" in ARGS
const PAPER = joinpath(@__DIR__, "output", EVENING ? "paper_evening" : "paper")
const PROFILE = "a1"
const COAL_WINDOW = EVENING ? (60:64) : (72:76)
const RESERVE_EXAMPLE = EVENING ? ("c2", 62) : ("f1", 74)
const INFEAS = "c1"
const INFEAS_STEPS = 2
RESIL.tau = 4

OUTAGE.enabled = true
OUTAGE.start, OUTAGE.stop = EVENING ? (63, 83) : (49, 72)

# Chosen buildings renumbered 1..N, with the current step measured and the later steps forecast
function load_subset()
    all_bs, buy, sell, _ = MPC_load_from_CSV(maximum(IDS), NUM_STEPS, DATA_DIR)
    measured(pred, act) = (p = copy(pred); p[:, 1] .= act; p)
    bs = [MPC_Building(b.loc, measured(b.pred_cons, b.act_cons), measured(b.pred_prod, b.act_prod), b.act_cons, b.act_prod,
                       b.max_storage, b.storage_max_flow, b.charge_eff, b.discharge_eff, i, zeros(NUM_STEPS))
          for (i, b) in enumerate(all_bs[IDS])]
    for b in bs
        b.SoC[1] = SOC0 * Float64(b.max_storage)
    end
    return bs, buy, sell
end

# Every building alone (D0, D1), with the outage problem during an outage
function decentralised(bs, max_coal_size, k, num_look_ahead, receding_horizon = false)
    agents = Vector(1:length(bs))
    outs = [in_outage(k) ? (single_optimise_outage(opt, bs[a], k, num_look_ahead, receding_horizon)[2], 1) :
            single_optimise_ADMM(opt, bs[a], k, num_look_ahead, receding_horizon) for a in agents]
    return agents, [o[1] for o in outs], sum(o[2] for o in outs)
end

# All buildings in one coalition (D2), with the outage problem during an outage
function centralised(bs, max_coal_size, k, num_look_ahead, receding_horizon = false)
    solve = in_outage(k) ? single_optimise_outage : single_optimise
    return [collect(eachindex(bs))], [solve(opt, bs, k, num_look_ahead, receding_horizon)[2]], 1
end

# Closed-loop run of one case; a building without a plan keeps its battery idle and trades with the grid
function run_case(label, former; enabled, cmax = CMAX, order = :paper, exact = false, fix = false)
    RESIL.enabled = enabled
    FORM.order, FORM.exact_bound, FORM.fix_dissolve = order, exact, fix
    bs, buy_p, sell_p = load_subset()
    global opt = MPC_optimiser(buy_p', sell_p')
    statuses = Any[]
    fails = Any[]
    short_next = zeros(NUM_STEPS)
    short_plan = zeros(NUM_STEPS)
    step_time = zeros(NUM_STEPS)
    solves = zeros(NUM_STEPS)
    uc = fill(NaN, NUM_STEPS, length(IDS))
    ud = fill(NaN, NUM_STEPS, length(IDS))
    recorded = function (bs, mcs, k, na, rh)
        t1 = time()
        coal, vars, iters = former(bs, mcs, k, na, rh)
        step_time[k] = time() - t1
        solves[k] = iters
        for (c, v) in zip(coal, vars)
            m = owner_model(first(v[4]))
            members = Int.(agent_ids(c))
            push!(statuses, termination_status(m))
            solved = last(statuses) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            solved || push!(fails, (k = k, members = members, status = last(statuses), primal = primal_status(m)))
            for (j, i) in enumerate(members)
                uc[k, i] = solved ? value(m[:pos_delta_s][1, j]) : NaN
                ud[k, i] = solved ? -value(m[:neg_delta_s][1, j]) : NaN
            end
            if haskey(m, :resil_short) && !isempty(m[:resil_short]) && has_values(m)
                rs = value.(m[:resil_short])
                all(isfinite, rs) || continue
                short_next[k+1] += rs[2]
                short_plan[k] += sum(rs)
            end
        end
        return coal, vars, iters
    end

    t0 = time()
    cost, (buy, sell), solves_per_step, _, hist, (shed, curt) = coal_MPC(recorded, bs, cmax, H)
    extra = 0.0
    for f in fails, i in f.members
        (in_outage(f.k) || buy[f.k, i] != 0 || sell[f.k, i] != 0) && continue
        net = bs[i].act_prod[f.k] - bs[i].act_cons[f.k]
        extra += net > 0 ? -sell_p[f.k] * net : -buy_p[f.k] * net
    end

    return (label = label, cmax = cmax, bs = bs, cost = cost + extra, booked = cost, hist = hist, time = time() - t0,
            resil = enabled, shed = sum(shed), curt = sum(curt), shed_k = shed, curt_k = curt,
            soc = reduce(hcat, [b.SoC for b in bs]), step_time = step_time, solves = solves, solves_per_step = solves_per_step,
            bad = count(s -> !(s in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)), statuses), nsolves = length(statuses),
            nfail = sum((length(f.members) for f in fails if !in_outage(f.k)); init = 0),
            fails = fails, short_next = short_next, short_plan = short_plan, uc = uc, ud = ud, buy = buy)
end

# Coalition-steps of a run whose total stored energy at k+1 is below their reserve
function assess(r)
    bs = r.bs
    violations = 0
    below = Any[]
    for k in 1:NUM_STEPS-1
        in_outage(k) && continue
        for c in r.hist[k]
            m = Int.(agent_ids(c))
            g = sum(bs[i].SoC[k+1] for i in m) - stilde_coal(bs[m], k, 1, RESIL.tau)[2]
            if g < -1e-3 * sum(Float64(bs[i].max_storage) for i in m)
                violations += 1
                push!(below, (k = k, members = m, gap = g))
            end
        end
    end
    return violations, below
end

function warmup()
    RESIL.enabled = true
    bs, buy, sell = load_subset()
    global opt = MPC_optimiser(buy', sell')
    resilience_focussed_coals(bs, CMAX, 1, H)
    OUTAGE.enabled && resilience_focussed_coals(bs, CMAX, OUTAGE.start, H)
    return nothing
end

horizon(bs, k) = max(min(length(pred_consumption(bs[1], k)), min(NUM_STEPS - k + 1, H)), 1)
clock(k) = @sprintf("%02d:%02d", (k - 1) ÷ 4, 15 * ((k - 1) % 4))
coalition_of(r, k, i) = first(a for a in r.hist[k] if i in Int.(agent_ids(a)))

function coal_sizes(r)
    sz = zeros(Int, length(IDS), NUM_STEPS)
    for k in 1:NUM_STEPS, a in r.hist[k]
        m = Int.(agent_ids(a))
        sz[m, k] .= length(m)
    end
    return sz
end

# Stored energy, reserve and reachable energy of building l alone (D1) at its first step without a plan
function infeasible_detail(r, l)
    i = findfirst(==(l), LABELS)
    b = r.bs[i]
    kf = first(f.k for f in r.fails if i in f.members && !in_outage(f.k))
    hk = horizon([b], kf)
    need = stilde_coal([b], kf, hk, RESIL.tau)[2:hk]
    reach = [min(Float64(b.max_storage), r.soc[kf, i] + Float64(b.charge_eff) * Float64(b.storage_max_flow) * j) for j in 0:hk-1]
    return (i = i, k = kf, soc = r.soc[1:kf, i], need = need, reach = reach, capacity = Float64(b.max_storage))
end

# Data of the paper figures, saved to runs.jls so that --figures can redraw them
function paper_data(runs)
    bs = runs["RC"].bs
    own = [k == 1 || in_outage(k - 1) ? NaN : stilde_coal([bs[i]], k - 1, 1, RESIL.tau)[2] for k in 1:NUM_STEPS, i in eachindex(IDS)]
    return (labels = LABELS, ids = IDS, num_steps = NUM_STEPS, outage = (OUTAGE.start, OUTAGE.stop),
            capacity = [Float64(b.max_storage) for b in bs], own_reserve = own,
            demand = reduce(hcat, [b.act_cons[1:NUM_STEPS] for b in bs]), generation = reduce(hcat, [b.act_prod[1:NUM_STEPS] for b in bs]),
            buy = vec(opt.energy_cost)[1:NUM_STEPS], sell = vec(opt.energy_sale)[1:NUM_STEPS],
            runs = Dict(c => (soc = r.soc, shed = r.shed_k, curt = r.curt_k, fails = [(f.k, f.members) for f in r.fails], sizes = coal_sizes(r),
                              members = [[Int.(agent_ids(a)) for a in r.hist[k]] for k in 1:NUM_STEPS])
                        for (c, r) in runs))
end

function write_csv(name, header, cols)
    mkpath(joinpath(PAPER, "data"))
    open(joinpath(PAPER, "data", name * ".csv"), "w") do io
        println(io, join(header, ","))
        for r in eachindex(cols[1])
            println(io, join((string(c[r]) for c in cols), ","))
        end
    end
end

# Figures of the paper as SVG in Times and their data as CSV, after removing the old fig_* files
function paper_figures(fd)
    mkpath(joinpath(PAPER, "data"))
    for d in (PAPER, joinpath(PAPER, "data")), f in readdir(d; join = true)
        isfile(f) && startswith(basename(f), "fig_") && rm(f)
    end
    default(fontfamily = "times", guidefontsize = 8, tickfontsize = 7, legendfontsize = 7, titlefontsize = 8,
            linewidth = 1.1, framestyle = :box, grid = false, foreground_color_legend = nothing, background_color_legend = nothing)
    nb, ns = length(fd.labels), fd.num_steps
    hours = collect((0:ns-1) ./ 4)
    steps = collect(1:ns)
    in_out(k) = fd.outage[1] <= k <= fd.outage[2]
    out_col = [Int(in_out(k)) for k in steps]
    tout = [(fd.outage[1] - 1) / 4, fd.outage[2] / 4]
    outage!(p) = vline!(p, tout, color = :red, linestyle = :dash, linewidth = 1, label = "")
    xt = (xticks = 0:4:24, xlims = (0, 24))
    xt_blank = (xticks = (0:4:24, fill("", 7)), xlims = (0, 24))
    preview = get(ENV, "PAPER_PNG_PREVIEW", "")
    function save(fig, name)
        savefig(fig, joinpath(PAPER, name * ".svg"))
        isempty(preview) || savefig(plot(fig, dpi = 300), joinpath(preview, name * ".png"))
        println("saved ", joinpath(PAPER, name * ".svg"))
    end
    style = Dict("D0" => (:gray55, :dash), "D1" => (:black, :dot), "D2" => (:darkgreen, :dashdot), "RC" => (:royalblue, :solid))
    cases = ["D0", "D1", "D2", "RC"]
    nofail(c) = [(k, i) for (k, m) in fd.runs[c].fails for i in m if !in_out(k)]
    blank(x) = isnan(x) ? "" : string(x)

    j = findfirst(==(PROFILE), fd.labels)
    top = max(maximum(fd.demand[:, j]), maximum(fd.generation[:, j]))
    q1 = plot(; title = "Building $(PROFILE)", ylabel = "Energy (kWh)", ylims = (0, 1.35 * top), legend = :topleft, xt_blank...)
    plot!(q1, hours, fd.demand[:, j], label = "demand", color = :black, linetype = :steppost)
    plot!(q1, hours, fd.generation[:, j], label = "generation", color = :seagreen, linestyle = :dash, linetype = :steppost)
    q2 = plot(; ylabel = "Price (£/kWh)", xlabel = "Time (h)", ylims = (0, 0.7), legend = :topleft, legend_columns = 2, xt...)
    plot!(q2, hours, fd.buy, label = "buy", color = :black, linetype = :steppost)
    plot!(q2, hours, fd.sell, label = "sell", color = :royalblue, linestyle = :dash, linetype = :steppost)
    save(plot(q1, q2, layout = grid(2, 1), size = (300, 290)), "fig_profiles")
    write_csv("fig_profiles", ["step", "hour", "outage", "demand_$(PROFILE)", "generation_$(PROFILE)", "price_buy", "price_sell"],
              (steps, hours, out_col, fd.demand[:, j], fd.generation[:, j], fd.buy, fd.sell))

    lead, kx = RESERVE_EXAMPLE
    il = findfirst(==(lead), fd.labels)
    mem = first(c for c in fd.runs["RC"].members[kx] if il in c)

    pal = [:navy, :darkorange, :seagreen, :purple, :goldenrod]
    step_pal = ["#332288", "#999933", "#882255", "#44AA99", "#AA4499"]
    t0, t1 = (first(COAL_WINDOW) - 1) / 4, last(COAL_WINDOW) / 4
    gap = 0.035
    keep(c) = any(in(mem), c)
    function coalitions_figure(name, full)
        drawn(k) = [c for c in fd.runs["RC"].members[k] if full || keep(c)]
        shown = sort(unique(reduce(vcat, [c for k in COAL_WINDOW for c in drawn(k)])))
        row = Dict(i => r for (r, i) in enumerate(shown))
        ny = length(shown)
        lowest(c) = minimum(row[i] for i in c)
        f = plot(; xlims = (t0, t1), ylims = (0.5, ny + 0.5), yticks = (1:ny, fd.labels[shown]),
                 xticks = ((COAL_WINDOW .- 0.5) ./ 4, [@sprintf("%.2f", (k - 1) / 4) for k in COAL_WINDOW]),
                 xlabel = "Time (h)", ylabel = "Building", legend = false, size = (300, 75 + 14 * ny))
        rows = Tuple{Int,Float64,Int,String,String}[]
        for (s, k) in enumerate(COAL_WINDOW)
            xl, xr = (k - 1) / 4 + gap / 2, k / 4 - gap / 2
            plot!(f, Shape([xl, xr, xr, xl], [0.5, 0.5, ny + 0.5, ny + 0.5]), fillcolor = :gray93, linewidth = 0, linealpha = 0)
            cs = sort(drawn(k), by = lowest)
            for (m, c) in enumerate(cs)
                x = xl + (xr - xl) * (m - 0.5) / length(cs)
                col = full ? step_pal[mod1(s, length(step_pal))] : pal[mod1(m, length(pal))]
                ys = [row[i] for i in c]
                if length(c) > 1
                    plot!(f, [x, x], [minimum(ys), maximum(ys)], color = col, linewidth = 1.3)
                    scatter!(f, fill(x, length(ys)), ys, color = col, markersize = 3.0, markerstrokewidth = 0)
                else
                    scatter!(f, [x], ys, color = :white, markerstrokecolor = col, markersize = 3.0, markerstrokewidth = 0.9)
                end
                push!(rows, (k, (k - 1) / 4, m, join(fd.labels[sort(c)], "+"), string(col)))
            end
        end
        vline!(f, filter(t -> t0 <= t <= t1, tout), color = :red, linestyle = :dash, linewidth = 1)
        save(f, name)
        cols = full ? (1:5) : (1:4)
        write_csv(name, ["step", "hour", "coalition", "members", "colour"][cols], Tuple(getindex.(rows, c) for c in cols))
    end
    coalitions_figure("fig_coalitions", false)
    coalitions_figure("fig_coalitions_all", true)

    ls = [il; sort(setdiff(mem, [il]))]
    panels = Any[]
    for (n, i) in enumerate(ls)
        top = (n == 1 ? 1.65 : 1.35) * max(fd.capacity[i], maximum(fd.demand[:, i]), n == 1 ? maximum(fd.generation[:, i]) : 0.0)
        p = plot(; ylabel = "Energy (kWh)", xlabel = n == length(ls) ? "Time (h)" : "", ylims = (0, top),
                 legend = n == 1 ? :topleft : false, legend_columns = 2, (n == length(ls) ? xt : xt_blank)...,
                 top_margin = n == 1 ? 0Plots.mm : -2.5Plots.mm, bottom_margin = n == length(ls) ? 0Plots.mm : -2.5Plots.mm)
        plot!(p, hours, fd.runs["RC"].soc[:, i], label = "stored energy", color = :royalblue, linetype = :steppost)
        plot!(p, hours, fd.own_reserve[:, i], label = "own reserve", color = :darkorange, linestyle = :dashdot, linetype = :steppost)
        plot!(p, hours, fd.demand[:, i], label = "demand", color = :black, linewidth = 0.8, linetype = :steppost)
        n == 1 && plot!(p, hours, fd.generation[:, i], label = "generation", color = :seagreen, linestyle = :dash,
                        linewidth = 0.8, linetype = :steppost)
        outage!(p)
        push!(panels, p)
    end
    np = length(ls)
    save(plot(panels..., layout = grid(np, 1, heights = [1.3; ones(np - 1)] ./ (np + 0.3)), size = (300, 95 * np + 45)),
         "fig_reserve")
    header, cols = ["step", "hour", "outage", "coalition_step", "generation_$(lead)"],
                   Any[steps, hours, out_col, [Int(k == kx) for k in steps], fd.generation[:, il]]
    for i in ls
        push!(header, "stored_$(fd.labels[i])", "own_reserve_$(fd.labels[i])", "demand_$(fd.labels[i])")
        push!(cols, fd.runs["RC"].soc[:, i], blank.(fd.own_reserve[:, i]), fd.demand[:, i])
    end
    write_csv("fig_reserve", header, Tuple(cols))

    d1f = nofail("D1")
    f = plot(; ylims = (0.4, nb + 0.6), yticks = (1:nb, fd.labels), xlabel = "Time (h)", ylabel = "Building", legend = false,
             size = (300, 220), xt...)
    scatter!(f, [(k - 1) / 4 + 0.125 for (k, _) in d1f], [i for (_, i) in d1f], marker = :rect, markersize = 2.2,
             markerstrokewidth = 0, color = :black)
    outage!(f)
    save(f, "fig_infeasible")
    write_csv("fig_infeasible", ["step", "hour", "building", "data_set_id"],
              ([k for (k, _) in d1f], [(k - 1) / 4 for (k, _) in d1f], [fd.labels[i] for (_, i) in d1f], [fd.ids[i] for (_, i) in d1f]))

    soc_tot = Dict(c => vec(sum(fd.runs[c].soc, dims = 2)) for c in cases)
    shed_tot = Dict(c => vec(sum(fd.runs[c].shed, dims = 2)) for c in cases)
    has_curt = all(c -> haskey(fd.runs[c], :curt), cases)
    curt_tot = has_curt ? Dict(c => vec(sum(fd.runs[c].curt, dims = 2)) for c in cases) : Dict{String,Vector{Float64}}()
    p1 = plot(; ylabel = "Energy (kWh)", ylims = (0, 1.45 * maximum(maximum.(values(soc_tot)))), legend = :topleft,
              legend_columns = 2, bottom_margin = -2.5Plots.mm, xt_blank...)
    p2 = plot(; ylabel = "Load shed (kWh)", xlabel = has_curt ? "" : "Time (h)", legend = false, top_margin = -2.5Plots.mm,
              bottom_margin = has_curt ? -2.5Plots.mm : 0Plots.mm, (has_curt ? xt_blank : xt)...)
    p3 = plot(; ylabel = "Curtailed (kWh)", xlabel = "Time (h)", legend = false, top_margin = -2.5Plots.mm, xt...)
    for c in cases
        col, ls = style[c]
        plot!(p1, hours, soc_tot[c], label = c, color = col, linestyle = ls, linetype = :steppost)
        plot!(p2, hours, shed_tot[c], label = c, color = col, linestyle = ls, linetype = :steppost)
        has_curt && plot!(p3, hours, curt_tot[c], label = c, color = col, linestyle = ls, linetype = :steppost)
    end
    foreach(outage!, has_curt ? (p1, p2, p3) : (p1, p2))
    if has_curt
        save(plot(p1, p2, p3, layout = grid(3, 1, heights = [0.38, 0.31, 0.31]), size = (300, 420)), "fig_energy_shed")
    else
        save(plot(p1, p2, layout = grid(2, 1, heights = [0.55, 0.45]), size = (300, 280)), "fig_energy_shed")
    end
    write_csv("fig_energy_shed", ["step", "hour", "outage", ("stored_$(c)" for c in cases)..., ("shed_$(c)" for c in cases)...,
                                  (has_curt ? ["curtailed_$(c)" for c in cases] : String[])...],
              (steps, hours, out_col, (soc_tot[c] for c in cases)..., (shed_tot[c] for c in cases)...,
               (has_curt ? [curt_tot[c] for c in cases] : Vector{Float64}[])...))
    return nothing
end

# Numbers of the text in results.txt and the rows of the two tables in table_*.tex
function paper_numbers(runs, sweep)
    bs = runs["RC"].bs
    cases = ["D0", "D1", "D2", "RC"]
    outk = OUTAGE.start:OUTAGE.stop
    normal = [k for k in 2:NUM_STEPS if !in_outage(k - 1)]
    largest(c, r) = c == "D2" ? length(IDS) : c in ("D0", "D1") ? 1 : r.cmax
    cap_str(x) = isinteger(x) ? string(Int(x)) : rstrip(rstrip(@sprintf("%.2f", x), '0'), '.')
    deficit = [i for i in eachindex(IDS) if sum(bs[i].act_cons[outk]) > sum(bs[i].act_prod[outk])]
    rows = [[(c, runs[c]) for c in cases]; [("RC", r) for r in sweep]]
    sort!(rows, by = x -> (findfirst(==(x[1]), cases), x[2].cmax))

    open(joinpath(PAPER, "table_buildings.tex"), "w") do io
        println(io, "% Building & peak demand (kWh) & peak generation (kWh) & capacity (kWh) & eta_c & eta_d, by test_n_buildings.jl")
        for (i, b) in enumerate(bs)
            @printf(io, "%s & %.1f & %.1f & %s & %.3f & %.3f\\\\\n", LABELS[i], maximum(b.act_cons[1:NUM_STEPS]),
                    maximum(b.act_prod[1:NUM_STEPS]), cap_str(Float64(b.max_storage)), Float64(b.charge_eff), Float64(b.discharge_eff))
        end
    end
    open(joinpath(PAPER, "table_results.tex"), "w") do io
        println(io, "% Case & cost & shed (kWh) & curtailed (kWh) & infeasibility exists (a step without a feasible plan), by test_n_buildings.jl")
        for (c, r) in rows
            name = c == "RC" ? "RC, \$\\mathcal{C}_{\\max}=$(r.cmax)\$" : c
            @printf(io, "%s & \$%.1f\$ & %.1f & %.1f & %s\\\\\n", name, r.cost, r.shed, r.curt,
                    r.nfail > 0 ? "\\checkmark" : "\$\\times\$")
        end
    end

    open(joinpath(PAPER, "results.txt"), "w") do io
        println(io, "Numbers of Section IV of eccpaper.tex, written by test_n_buildings.jl")
        println(io, "buildings: ", join(("$(LABELS[i]) = id $(IDS[i])" for i in eachindex(IDS)), ", "))
        @printf(io, "Cmax %d, K %d, tau %d, outage steps %d-%d (%s to %s), max over buildings of 1 - eta_c eta_d %.3f\n\n",
                CMAX, H, RESIL.tau, OUTAGE.start, OUTAGE.stop, clock(OUTAGE.start), clock(OUTAGE.stop + 1),
                maximum(1 - Float64(b.charge_eff) * Float64(b.discharge_eff) for b in bs))
        println(io, "case | largest coalition | cost | cost booked by coal_MPC | shed | curtailed | steps without a plan, summed ",
                "over buildings | mean, max time per step (s) | solves per step | solves not OPTIMAL")
        for (c, r) in rows
            @printf(io, "%s | %d | %.2f | %.2f | %.2f | %.2f | %d | %.3f, %.3f | %.1f | %d of %d\n", c, largest(c, r), r.cost,
                    r.booked, r.shed, r.curt, r.nfail, mean(r.step_time), maximum(r.step_time), mean(r.solves), r.bad, r.nsolves)
        end
        println(io, "\ntotal stored energy at the start and at the end of the outage, load shed of the buildings with a deficit ",
                "during the outage (", join(LABELS[deficit], ", "), ")")
        for c in cases
            r = runs[c]
            @printf(io, "  %s: %.2f, %.2f | %.2f of %.2f\n", c, sum(r.soc[OUTAGE.start, :]), sum(r.soc[OUTAGE.stop + 1, :]),
                    sum(r.shed_k[:, deficit]), r.shed)
        end
        println(io, "\ndemand and generation of each building during the outage, and its load shed in D0, D1, D2, RC")
        for (i, b) in enumerate(bs)
            @printf(io, "  %-3s %8.1f %8.1f | %7.2f %7.2f %7.2f %7.2f\n", LABELS[i], sum(b.act_cons[outk]), sum(b.act_prod[outk]),
                    (sum(runs[c].shed_k[:, i]) for c in cases)...)
        end
        println(io, "\nD1, steps without a plan: building, first and last step, number, stored energy, reserve needed at k+1")
        for i in eachindex(IDS)
            ks = [f.k for f in runs["D1"].fails if i in f.members && !in_outage(f.k)]
            isempty(ks) && continue
            need = [stilde_coal([runs["D1"].bs[i]], k, 1, RESIL.tau)[2] for k in ks]
            @printf(io, "  %-3s %s-%s %2d steps, stored %.2f-%.2f of %s, reserve %.2f-%.2f\n", LABELS[i], clock(first(ks)),
                    clock(last(ks)), length(ks), minimum(runs["D1"].soc[ks, i]), maximum(runs["D1"].soc[ks, i]),
                    cap_str(Float64(bs[i].max_storage)), minimum(need), maximum(need))
        end
        println(io, "\nRC, steps of normal operation (", length(normal), ") at which a building holds less than its own reserve")
        for (i, b) in enumerate(bs)
            n = count(k -> runs["RC"].soc[k, i] < stilde_coal([b], k - 1, 1, RESIL.tau)[2] - 1e-3 * Float64(b.max_storage), normal)
            @printf(io, "  %-3s %2d\n", LABELS[i], n)
        end
        sz = coal_sizes(runs["RC"])
        ncoal = [length(runs["RC"].hist[k]) for k in 1:NUM_STEPS]
        println(io, "\nRC coalitions per step: mean ", round(mean(ncoal), digits = 2), ", min ", minimum(ncoal), ", max ", maximum(ncoal),
                "; mean size of the coalition of a building ", round(mean(sz), digits = 2))
        for (name, ks) in (("before the outage", 1:OUTAGE.start-1), ("during the outage", outk), ("after the outage", OUTAGE.stop+1:NUM_STEPS))
            @printf(io, "  %-18s buildings alone %.2f of %d on average\n", name, mean(count(==(1), sz[:, k]) for k in ks), length(IDS))
        end
        println(io, "\nsimultaneous charging and discharging in the applied plans, min(uc, ud) > 1e-3: building-steps, largest min")
        for c in ("D2", "RC"), (name, ks) in (("normal operation", [k for k in 1:NUM_STEPS if !in_outage(k)]), ("outage", collect(outk)))
            m = filter(!isnan, vec(min.(runs[c].uc[ks, :], runs[c].ud[ks, :])))
            @printf(io, "  %s, %-16s %d of %d, %.4f\n", c, name, count(>(1e-3), m), length(m), maximum(m; init = 0.0))
        end
        groups(k) = join((join(LABELS[g], "+") for g in sort([sort(Int.(agent_ids(a))) for a in runs["RC"].hist[k]], by = first)), "  ")
        d = infeasible_detail(runs["D1"], INFEAS)
        println(io, "\nD1, first step without a plan of $(INFEAS) (fig_infeasible_detail): stored energy, reachable margin, and ",
                "at the next steps of the horizon the reserve required by the plan and the largest reachable stored energy")
        @printf(io, "  %s at %s: stored %.2f of %s, reachable margin %.2f\n", INFEAS, clock(d.k), last(d.soc), cap_str(d.capacity),
                minimum(d.reach[2:end] .- d.need))
        println(io, "    required  ", join((@sprintf("%.2f", x) for x in d.need), " "))
        println(io, "    reachable ", join((@sprintf("%.2f", x) for x in d.reach[2:end]), " "))
        b = runs["RC"].bs[d.i]
        s_rc = runs["RC"].soc[d.k, d.i]
        reach_rc = [min(d.capacity, s_rc + Float64(b.charge_eff) * Float64(b.storage_max_flow) * j) for j in eachindex(d.need)]
        @printf(io, "  in RC at %s: stored %.2f, reachable margin if alone %.2f; coalitions of RC:\n", clock(d.k), s_rc,
                minimum(reach_rc .- d.need))
        for k in d.k-INFEAS_STEPS:d.k+INFEAS_STEPS
            @printf(io, "    %2d %s %s\n", k, clock(k), groups(k))
        end
        lead, kx = RESERVE_EXAMPLE
        il = findfirst(==(lead), LABELS)
        rc = runs["RC"]
        mem = sort(Int.(agent_ids(first(a for a in rc.hist[kx] if il in Int.(agent_ids(a))))))
        @printf(io, "\nfig_reserve: coalition %s of RC at %s; after this step it holds %.2f in total, its reserve is %.2f\n",
                join(LABELS[mem], "+"), clock(kx), sum(rc.soc[kx+1, mem]), stilde_coal(rc.bs[mem], kx, 1, RESIL.tau)[2])
        for i in mem
            @printf(io, "  %-3s stored %.2f, own reserve %.2f, capacity %s\n", LABELS[i], rc.soc[kx+1, i],
                    stilde_coal([rc.bs[i]], kx, 1, RESIL.tau)[2], cap_str(Float64(rc.bs[i].max_storage)))
        end
        println(io, "\noutage, per hour: demand and generation of all buildings | stored energy at the start of the hour, load ",
                "shed and generation curtailed in D0, D1, D2, RC")
        dem = [sum(b.act_cons[k] for b in bs) for k in 1:NUM_STEPS]
        gen = [sum(b.act_prod[k] for b in bs) for k in 1:NUM_STEPS]
        cols3(f) = join((@sprintf("%8.1f", f(runs[c])) for c in cases), "")
        for h in (OUTAGE.start - 1) ÷ 4 : OUTAGE.stop ÷ 4 - 1
            ks = 4h+1:4h+4
            @printf(io, "  %2d-%2d h | %7.1f %7.1f | stored%s | shed%s | curtailed%s\n", h, h + 1, sum(dem[ks]), sum(gen[ks]),
                    cols3(r -> sum(r.soc[first(ks), :])), cols3(r -> sum(r.shed_k[ks, :])), cols3(r -> sum(r.curt_k[ks, :])))
        end
        @printf(io, "  stored at %s:%s, total capacity %.2f\n", clock(OUTAGE.stop + 1), cols3(r -> sum(r.soc[OUTAGE.stop + 1, :])),
                sum(Float64(b.max_storage) for b in bs))
        k0 = OUTAGE.start - 1
        @printf(io, "  predicted at %s, over the window of the reserve at %s: demand %.1f, generation %.1f\n", clock(k0),
                clock(OUTAGE.start), sum(sum(b.pred_cons[k0, 3:2+RESIL.tau]) for b in bs), sum(sum(b.pred_prod[k0, 3:2+RESIL.tau]) for b in bs))
        println(io, "  RC coalitions at ", clock(k0), " and their reserve at ", clock(OUTAGE.start), ": ",
                join((join(LABELS[m], "+") * @sprintf(" %.2f", stilde_coal(runs["RC"].bs[m], k0, 1, RESIL.tau)[2])
                      for m in (sort(Int.(agent_ids(a))) for a in runs["RC"].hist[k0])), ", "))
        println(io, "\nRC coalitions at the steps of fig_coalitions")
        for k in COAL_WINDOW
            @printf(io, "  %2d %s%s %s\n", k, clock(k), in_outage(k) ? " outage" : "       ", groups(k))
        end
    end
    println("saved ", joinpath(PAPER, "results.txt"), ", table_buildings.tex and table_results.tex")
end

mkpath(PAPER)
if "--figures" in ARGS
    paper_figures(deserialize(joinpath(PAPER, "runs.jls")))
    exit()
end

warmup()
results = [
    run_case("decentralised, no reserve", decentralised; enabled = false),
    run_case("decentralised, with reserve", decentralised; enabled = true),
    run_case("centralised, with reserve", centralised; enabled = true),
    run_case("resilience, with reserve", resilience_focussed_coals; enabled = true)
]
sweep = [run_case("resilience, with reserve", resilience_focussed_coals; enabled = true, cmax = c) for c in CMAX_SWEEP]
FORM.order, FORM.exact_bound, FORM.fix_dissolve = :paper, false, false
checks = [assess(r) for r in results]
alone = results[findfirst(r -> r.label == "decentralised, with reserve", results)]
main = results[findfirst(r -> startswith(r.label, "resilience"), results)]

RESIL.enabled = true
needs = [[k for k in 1:NUM_STEPS if !in_outage(k) && reach_margin(main.bs[[i]], k, horizon(main.bs, k)) < -RES_TOL] for i in eachindex(IDS)]

outdir = joinpath(@__DIR__, "output", EVENING ? "n_buildings_evening" : "n_buildings")
mkpath(outdir)

table_file = joinpath(outdir, "$(OUT).txt")
ok = true
open(table_file, "w") do io
    @printf(io, "%-32s %8s | %8s %9s | %10s %9s %6s | %s\n", "case", "cost", "shed", "curtailed",
            "t_mean (s)", "t_max (s)", "solves", "not OPTIMAL")
    for (r, (viol, _)) in zip(results, checks)
        startswith(r.label, "resilience") && (viol > 0 || r.bad > 0) && (global ok = false)
        @printf(io, "%-32s %8.2f | %8.2f %9.2f | %10.3f %9.3f %6.1f | %4d of %4d\n", r.label, r.cost,
                r.shed, r.curt, sum(r.step_time) / NUM_STEPS, maximum(r.step_time), r.solves_per_step,
                r.bad, r.nsolves)
    end
    println(io)
    for (col, text) in [
            "cost"        => "realised grid cost, with the trade of the buildings without a plan",
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

hours = (0:NUM_STEPS-1) ./ 4
on = [i for (i, r) in enumerate(results) if r.resil]
colours = (:gray, :black, :royalblue)
outage_lines!(p; label = "") = OUTAGE.enabled &&
    vline!(p, [(OUTAGE.start - 1) / 4, OUTAGE.stop / 4], color = :red, ls = :dash, lw = 1.5, label = label)
function curve(r, i)
    soc = copy(r.bs[i].SoC)
    ks = [f.k for f in r.fails if i in f.members]
    startswith(r.label, "decentralised") && !isempty(ks) && (soc[first(ks)+1:end] .= NaN)
    return soc
end

combinedir = joinpath(outdir, "combine")
mkpath(combinedir)

for (f, group) in enumerate(Iterators.partition(eachindex(IDS), 3))
    panels = Any[]
    for i in group
        p = plot(title = NAMES[i], ylabel = "energy", xlabel = i == last(group) ? "hour of day" : "",
                 legend = i == first(group) ? :topright : false)
        for (j, col) in zip(on, colours)
            plot!(p, hours, curve(results[j], i), label = results[j].label, lw = 2, color = col, linetype = :steppost)
        end
        b = results[on[1]].bs[i]
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

singledir = joinpath(outdir, "single")
mkpath(singledir)
upto(n) = [hours[1:n]; n / 4]
headroom(v) = (lo = minimum(filter(isfinite, v)); hi = maximum(filter(isfinite, v)); (lo - 0.05 * (hi - lo), hi + 0.6 * (hi - lo) + (hi == lo)))
for i in eachindex(IDS)
    ks = [f.k for f in alone.fails if i in f.members]
    isempty(ks) && continue
    kf, b = first(ks), alone.bs[i]
    xr = (0, kf / 4)
    p1 = plot(title = NAMES[i] * " alone", ylabel = "energy", legend = :topleft, xlims = xr)
    p2 = plot(ylabel = "energy", legend = :topleft, xlims = xr)
    p3 = plot(ylabel = "energy", xlabel = "hour of day", legend = :topleft, xlims = xr)
    if kf > 1
        n = kf - 1
        plot!(p1, upto(kf), [b.SoC[1:kf]; b.SoC[kf]], label = "stored energy", lw = 2, color = :gray, linetype = :steppost)
        own = [k == 1 || in_outage(k - 1) ? NaN : stilde_coal([b], k - 1, 1, RESIL.tau)[2] for k in 1:kf+1]
        plot!(p1, upto(kf), own, label = "own tilde_s", lw = 1.5, color = :darkorange, ls = :dash, linetype = :steppost)
        plot!(p2, upto(n), [alone.uc[1:n, i]; alone.uc[n, i]], label = "charge", lw = 2, color = :seagreen, linetype = :steppost)
        plot!(p2, upto(n), [alone.ud[1:n, i]; alone.ud[n, i]], label = "discharge", lw = 2, color = :firebrick, linetype = :steppost)
        plot!(p3, upto(n), [alone.buy[1:n, i]; alone.buy[n, i]], label = "bought from the grid", lw = 2, color = :royalblue,
              linetype = :steppost)
        plot!(p1, ylims = headroom([b.SoC[1:kf]; own]))
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

coal_file = joinpath(combinedir, "$(OUT)_coalitions.txt")
open(coal_file, "w") do io
    println(io, "coalitions of the run \"$(main.label)\", Cmax $(CMAX); building ids, a + joins the members of a coalition")
    for k in 1:NUM_STEPS
        groups = sort([sort(IDS[Int.(agent_ids(a))]) for a in main.hist[k]], by = first)
        @printf(io, "%2d  %s%s  %s\n", k, clock(k), in_outage(k) ? "  outage" : "        ", join(join.(groups, "+"), "  "))
    end
end
println("saved ", coal_file)

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

paper_runs = Dict("D0" => results[1], "D1" => results[2], "D2" => results[3], "RC" => main)
paper_numbers(paper_runs, sweep)
fd = paper_data(paper_runs)
serialize(joinpath(PAPER, "runs.jls"), fd)
paper_figures(fd)
