using JuMP, SCS
using LinearAlgebra

include("Buildings.jl")
struct MPC_optimiser
    energy_cost::Array{Float64}
    energy_sale::Array{Float64}
end

function energy_cost_k(opt::MPC_optimiser, k::Int, num_steps::Int=96)
	last_elem = length(opt.energy_cost)
	return vcat(opt.energy_cost[k:last_elem], opt.energy_cost[1:k-1])[1:num_steps]
end

function energy_sale_k(opt::MPC_optimiser, k::Int, num_steps::Int=96)
	last_elem = length(opt.energy_sale)
	return vcat(opt.energy_sale[k:last_elem], opt.energy_sale[1:k-1])[1:num_steps]
end

Base.@kwdef mutable struct ResilienceParams
    enabled::Bool = true   # off = the paper's LP, unchanged
    tau::Int = 4           # reserve window tau of eq. (19): net demand of steps t..t+tau (m = 0..tau)
    rho::Float64 = 10.0    # soft var penalty 
end
const RESIL = ResilienceParams()

# Outage management after paper/SEGAN-D-26-04924.pdf, Section 3.3: no upstream grid from step OUTAGE.start to
# OUTAGE.stop (both included). A closed loop calls single_optimise_outage instead of the scheduling problem at the steps
# where in_outage(k) is true (test_single_optimise_soc.jl does; coal_MPC does not).
Base.@kwdef mutable struct OutageParams
    enabled::Bool = false          # off = grid available at every step
    start::Int = 21                # first step without grid (step 21 = 05:00, SEGAN's fault from the 5th hour)
    stop::Int = 48                 # last step without grid (step 48 = 11:45-12:00)
    keep_coalitions::Bool = true   # for a closed loop with coalitions: keep those of step start-1 (not used anywhere yet)
    c_shed::Float64 = 1.0          # c^shed in (29a), per unit of load not supplied
    c_curt::Float64 = 0.1          # c^curt in (29a), per unit of PV curtailed
    eps::Float64 = 0.05            # epsilon in (29a), per unit of charging; must exceed c_curt*(1 - eta_c*eta_d), or the LP
                                   # burns surplus PV by charging and discharging at once instead of curtailing it
end
const OUTAGE = OutageParams()
in_outage(k::Int) = OUTAGE.enabled && OUTAGE.start <= k <= OUTAGE.stop


function stilde_coal(bs::Vector{MPC_Building}, k::Int, H::Int, tau::Int)
    cap = sum(Float64(b.max_storage) for b in bs)
    st = zeros(H + 1)
    for t in 1:H+1
        win = t+1:min(t + tau, size(bs[1].pred_cons, 2))
        st[t] = max(0, min(sum(sum(b.pred_cons[k, win] .- b.pred_prod[k, win]) for b in bs), cap))
    end
    return st
end



function optimise(opt::MPC_optimiser,bs::Vector{Building},ADMM::Bool=true,receding_horizon::Bool=false)
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = length(consumption(bs[1]))
	set_silent(model)
	
	max_flow_val = repeat(hcat([max_flow(b) for b in bs])', num_steps, 1)
	@variable(model, 0<=pos_delta_s[t=1:num_steps, b=1:num_builds]<=max_flow_val[t,b])
	@variable(model, -max_flow_val[t,b]<=neg_delta_s[t=1:num_steps, b=1:num_builds]<=0)
	@variable(model, delta_s[1:num_steps,1:num_builds])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)
	
	@variable(model, grid_cons[1:num_steps, 1:num_builds] >= 0)
	@variable(model, grid_sell[1:num_steps, 1:num_builds] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:num_steps, 1:num_builds])
	capacities = repeat(hcat([max_store(b) for b in bs])', num_steps, 1)
	#println(capacities)
	#println(capacities.size)
	@variable(model, 0 <= charge[t=1:num_steps, b=1:num_builds] <= capacities[t,b])
	@constraint(model, init_charge, charge[1,:]==[0 for b in bs])
	@variable(model, costs[1:num_builds])

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))

	#println(charge_eff_vec)

	charge_eff_vec = hcat([charge_eff(b) for b in bs])
	discharge_eff_vec = hcat([1/discharge_eff(b) for b in bs])
	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec'))

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])
	
	consumps = reduce(hcat, [consumption(b) for b in bs])
	prods = reduce(hcat,[production(b) for b in bs])
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, costs'.==opt.energy_cost*grid_cons-opt.energy_sale*grid_sell)

	@objective(model, Min, ones(num_builds)'*costs)
	
	optimize!(model)
	#println(sum(value(neg_delta_s.*pos_delta_s)))

	return sum(value(costs)), [delta_s, grid_cons, grid_sell, charge, costs], 1
end

function optimise(opt::MPC_optimiser,b::Building,ADMM::Bool=true,receding_horizon::Bool=false)
	return optimise(opt,[b])
end

function single_optimise(opt::MPC_optimiser,bs::Vector{MPC_Building},k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = num_look_ahead
	if !receding_horizon
		num_steps = min(length(bs[1].act_cons)-k+1,num_steps)
	end
	num_steps = min(length(pred_consumption(bs[1],k)),num_steps)
	num_steps = max(num_steps, 1)
	set_silent(model)
	# P3.5: Tighten SCS tolerance from 1e-2 to 1e-4 for cross-machine
	# reproducibility of poss_coal_vals (the 1e-2 tolerance let near-tied
	# pair values flip argmax between machines).
	set_optimizer_attribute(model, "eps_abs", 1e-4)
	set_optimizer_attribute(model, "eps_rel", 1e-4)
	
	max_flow_val = repeat(hcat([max_flow(b) for b in bs])', num_steps, 1)
	@variable(model, 0<=pos_delta_s[t=1:num_steps, b=1:num_builds]<=max_flow_val[t,b])
	@variable(model, -max_flow_val[t,b]<=neg_delta_s[t=1:num_steps, b=1:num_builds]<=0)
	@variable(model, delta_s[1:num_steps,1:num_builds])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)
	
	@variable(model, grid_cons[1:num_steps, 1:num_builds] >= 0)
	@variable(model, grid_sell[1:num_steps, 1:num_builds] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:num_steps, 1:num_builds])
	capacities = repeat(hcat([max_store(b) for b in bs])', num_steps, 1)
	#println(capacities)
	#println(capacities.size)
	@variable(model, 0 <= charge[t=1:num_steps, b=1:num_builds] <= capacities[t,b])
	@variable(model, costs[1:num_builds])

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))

	#println(charge_eff_vec)

	charge_eff_vec = hcat([charge_eff(b) for b in bs])
	discharge_eff_vec = hcat([1/discharge_eff(b) for b in bs])

	init_charge_mat = vcat([b.SoC[k] for b in bs]',zeros(num_steps-1,num_builds))
	# println([b.SoC[k] for b in bs])
	# if isnan(bs[1].SoC[k][1])
	# 	b=bs[1]
	# 	println(b)
	# 	println(b.SoC)
	# 	println(k)
	# end
	# println(charge_mat)
	#println([b.SoC[k] for b in bs])
	#println(k)
	# println(init_charge_mat)
	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec')+init_charge_mat)

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])
	
	consumps = reduce(hcat, [pred_consumption(b,k,num_steps) for b in bs])
	prods = reduce(hcat,[pred_production(b,k,num_steps) for b in bs])
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, costs'.==energy_cost_k(opt,k,num_steps)'*grid_cons-energy_sale_k(opt,k,num_steps)'*grid_sell) # need to fix this

	shortfall_cost = 0.0
	if RESIL.enabled
		st = stilde_coal(bs, k, num_steps, RESIL.tau)
		@constraint(model, resil_floor[t = 2:num_steps], sum(charge[t, b] for b in 1:num_builds) >= st[t])


		# @variable(model, resil_short[t = 2:num_steps] >= 0)
		# @constraint(model, resil_floor[t = 2:num_steps], sum(charge[t, b] for b in 1:num_builds) + resil_short[t] >= st[t])
		# shortfall_cost = RESIL.rho * sum((resil_short[t] for t in 2:num_steps); init = 0.0)

		
		# soc_end = [charge[num_steps, b] + Float64(charge_eff(bs[b])) * pos_delta_s[num_steps, b] +
		#            neg_delta_s[num_steps, b] / Float64(discharge_eff(bs[b])) for b in 1:num_builds]
		# @constraint(model, resil_end, sum(soc_end) >= st[num_steps + 1])
	end
	@objective(model, Min, ones(num_builds)'*costs)
	# @objective(model, Min, ones(num_builds)'*costs + shortfall_cost)
	
	optimize!(model)
	#println(sum(value(neg_delta_s.*pos_delta_s)))

	return model, [delta_s, grid_cons, grid_sell, charge, costs, coal_exch]
end

# Outage management problem for the coalition bs at step k
#   - no power exchange with the upstream grid (21): grid_cons and grid_sell are dropped;
#   - load can be shed and PV curtailed (24)-(25): served_cons <= consumps, used_prod <= prods;
#   - the cost (29a) is c_shed * shed + c_curt * curtailment + eps * charging instead of the electricity bill.
function single_optimise_outage(opt::MPC_optimiser,bs::Vector{MPC_Building},k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = num_look_ahead
	if !receding_horizon
		num_steps = min(length(bs[1].act_cons)-k+1,num_steps)
	end
	num_steps = min(length(pred_consumption(bs[1],k)),num_steps)
	num_steps = max(num_steps, 1)
	set_silent(model)
	set_optimizer_attribute(model, "eps_abs", 1e-4)
	set_optimizer_attribute(model, "eps_rel", 1e-4)

	max_flow_val = repeat(hcat([max_flow(b) for b in bs])', num_steps, 1)
	@variable(model, 0<=pos_delta_s[t=1:num_steps, b=1:num_builds]<=max_flow_val[t,b])
	@variable(model, -max_flow_val[t,b]<=neg_delta_s[t=1:num_steps, b=1:num_builds]<=0)
	@variable(model, delta_s[1:num_steps,1:num_builds])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)

	consumps = reduce(hcat, [pred_consumption(b,k,num_steps) for b in bs])
	prods = reduce(hcat,[pred_production(b,k,num_steps) for b in bs])
	@variable(model, 0 <= served_cons[t=1:num_steps, b=1:num_builds] <= consumps[t,b])   # load actually supplied
	@variable(model, 0 <= used_prod[t=1:num_steps, b=1:num_builds] <= prods[t,b])        # PV actually used

	@variable(model, coal_exch[1:num_steps, 1:num_builds])
	capacities = repeat(hcat([max_store(b) for b in bs])', num_steps, 1)
	@variable(model, 0 <= charge[t=1:num_steps, b=1:num_builds] <= capacities[t,b])
	@variable(model, costs[1:num_builds])

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))
	charge_eff_vec = hcat([charge_eff(b) for b in bs])
	discharge_eff_vec = hcat([1/discharge_eff(b) for b in bs])
	init_charge_mat = vcat([b.SoC[k] for b in bs]',zeros(num_steps-1,num_builds))
	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec')+init_charge_mat)

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])

	@constraint(model, power_c, served_cons+delta_s+coal_exch.==used_prod)   # (21): no grid_cons, grid_sell

	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)

	# (29a), with shed = consumps - served_cons (23) and curtailment = prods - used_prod (22)
	@constraint(model, cost_c, costs'.==OUTAGE.c_shed*ones(num_steps)'*(consumps-served_cons)+OUTAGE.c_curt*ones(num_steps)'*(prods-used_prod)+OUTAGE.eps*ones(num_steps)'*pos_delta_s)

	@objective(model, Min, ones(num_builds)'*costs)

	optimize!(model)

	return model, [delta_s, nothing, nothing, charge, costs, coal_exch, served_cons, used_prod]
end

function ADMM_build_opt(b::MPC_Building,coal_trades::Vector{Float64},lambda::Vector{Float64},k::Int,c::Float64)
	model = Model(SCS.Optimizer)
	num_steps = length(lambda)
	set_silent(model)
	
	max_flow_val = ones(num_steps).*max_flow(b)
	@variable(model, 0<=pos_delta_s[t=1:num_steps]<=max_flow_val[t])
	@variable(model, -max_flow_val[t]<=neg_delta_s[t=1:num_steps]<=0)
	@variable(model, delta_s[1:num_steps])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)
	
	@variable(model, grid_cons[1:num_steps] >= 0)
	@variable(model, grid_sell[1:num_steps] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:num_steps])
	capacities = ones(num_steps).*max_store(b)
	#println(capacities)
	#println(capacities.size)
	@variable(model, 0 <= charge[t=1:num_steps] <= capacities[t])
	@variable(model, cost)

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))

	#println(charge_eff_vec)

	charge_eff_vec = charge_eff(b)'
	discharge_eff_vec = 1/discharge_eff(b)'

	init_charge_mat = vcat(b.SoC[k]',zeros(num_steps-1,1))

	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec')+init_charge_mat)

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])
	
	consumps =  pred_consumption(b,k,num_steps)
	prods =pred_production(b,k,num_steps)
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	# @constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, cost==energy_cost_k(opt,k,num_steps)'*grid_cons-energy_sale_k(opt,k,num_steps)'*grid_sell) # need to fix this

	@objective(model, Min, cost+lambda'*coal_exch+(c/2)*(coal_exch-coal_trades)'*(coal_exch-coal_trades))
	
	optimize!(model)
	#println(sum(value(neg_delta_s.*pos_delta_s)))
	return model, [delta_s, grid_cons, grid_sell, charge, cost, coal_exch]
end

function ADMM_coal_update(lambda::Vector{Vector{Float64}},proposed_coals::Vector{Vector{Float64}},c::Float64)
	model = Model(SCS.Optimizer)
	num_builds = length(lambda)
	num_steps = length(lambda[1])
	set_silent(model)
	@variable(model, coal_exch[1:num_steps,1:num_builds])
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)

	proposed_coals = reduce(hcat, proposed_coals)
	lambda = reduce(hcat, lambda)
	@objective(model, Min,-sum(lambda.*coal_exch)+(c/2)*sum((proposed_coals-coal_exch).*(proposed_coals-coal_exch)))

	optimize!(model)

	return model, coal_exch
end

# P3.1: Model-reusing variants. Build JuMP model once per (building, k, horizon),
# then on each ADMM iteration only the objective parameters (lambda, coal_trades)
# are updated via set_parameter_value. SCS reuses its internal factorisation.

struct ADMMBuildCache
	model::Model
	delta_s::Vector{VariableRef}
	grid_cons::Vector{VariableRef}
	grid_sell::Vector{VariableRef}
	charge::Vector{VariableRef}
	cost::VariableRef
	coal_exch::Vector{VariableRef}
	lam::Vector{VariableRef}
	ct::Vector{VariableRef}
end

function ADMM_build_opt_init(b::MPC_Building, k::Int, num_steps::Int)
	model = Model(SCS.Optimizer)
	set_silent(model)
	# P3.2: Relax SCS tolerance — the outer ADMM already converges to 1e-5,
	# so each per-iteration SCS solve doesn't need to be tight. Cuts SCS iters ~3x.
	# P3.5: Tightened from 1e-2 to 1e-4 for cross-machine reproducibility.
	set_optimizer_attribute(model, "eps_abs", 1e-4)
	set_optimizer_attribute(model, "eps_rel", 1e-4)

	max_flow_val = ones(num_steps).*max_flow(b)
	@variable(model, 0<=pos_delta_s[t=1:num_steps]<=max_flow_val[t])
	@variable(model, -max_flow_val[t]<=neg_delta_s[t=1:num_steps]<=0)
	@variable(model, delta_s[1:num_steps])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)

	@variable(model, grid_cons[1:num_steps] >= 0)
	@variable(model, grid_sell[1:num_steps] >= 0)

	@variable(model, coal_exch[1:num_steps])
	capacities = ones(num_steps).*max_store(b)
	@variable(model, 0 <= charge[t=1:num_steps] <= capacities[t])
	@variable(model, cost)

	# P3.1: lambda and coal_trades as Parameters so SCS reuses factorisation across iters
	@variable(model, lam[1:num_steps] in Parameter.(zeros(num_steps)))
	@variable(model, ct[1:num_steps] in Parameter.(zeros(num_steps)))

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))
	charge_eff_vec = charge_eff(b)'
	discharge_eff_vec = 1/discharge_eff(b)'
	init_charge_mat = vcat(b.SoC[k]',zeros(num_steps-1,1))

	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec')+init_charge_mat)
	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])

	consumps = pred_consumption(b,k,num_steps)
	prods = pred_production(b,k,num_steps)
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	@constraint(model, cost_c, cost==energy_cost_k(opt,k,num_steps)'*grid_cons-energy_sale_k(opt,k,num_steps)'*grid_sell)

	# Objective with parameters; c is fixed at 0.5 within an ADMM run
	c = 0.5
	@objective(model, Min, cost + lam'coal_exch + (c/2)*(coal_exch-ct)'*(coal_exch-ct))

	return ADMMBuildCache(model, delta_s, grid_cons, grid_sell, charge, cost, coal_exch, lam, ct)
end

function ADMM_build_opt_solve(cache::ADMMBuildCache, coal_trades::Vector{Float64}, lambda::Vector{Float64}, c::Float64)
	# Update only parameter values; model structure and SCS factorisation are reused
	for i in eachindex(lambda)
		set_parameter_value(cache.lam[i], lambda[i])
		set_parameter_value(cache.ct[i], coal_trades[i])
	end
	# P3.2: Warm-start SCS from the previous primal solution to cut solver iterations
	if has_values(cache.model)
		for v in all_variables(cache.model)
			set_start_value(v, value(v))
		end
	end
	optimize!(cache.model)
	return [cache.delta_s, cache.grid_cons, cache.grid_sell, cache.charge, cache.cost, cache.coal_exch]
end

struct ADMMCoalCache
	model::Model
	coal_exch::Matrix{VariableRef}
	lam::Matrix{VariableRef}
	prop::Matrix{VariableRef}
end

function ADMM_coal_update_init(num_builds::Int, num_steps::Int)
	model = Model(SCS.Optimizer)
	set_silent(model)
	# P3.2: Relax SCS tolerance (matches the per-building subproblem)
	# P3.5: Tightened from 1e-2 to 1e-4 for cross-machine reproducibility.
	set_optimizer_attribute(model, "eps_abs", 1e-4)
	set_optimizer_attribute(model, "eps_rel", 1e-4)
	@variable(model, coal_exch[1:num_steps,1:num_builds])
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	# Parameters for lambda and proposed_coals
	@variable(model, lam[1:num_steps,1:num_builds] in Parameter.(zeros(num_steps, num_builds)))
	@variable(model, prop[1:num_steps,1:num_builds] in Parameter.(zeros(num_steps, num_builds)))
	c = 0.5
	@objective(model, Min, -sum(lam.*coal_exch)+(c/2)*sum((prop-coal_exch).*(prop-coal_exch)))
	return ADMMCoalCache(model, coal_exch, lam, prop)
end

function ADMM_coal_update_solve(cache::ADMMCoalCache, lambda::Vector{Vector{Float64}}, proposed_coals::Vector{Vector{Float64}}, c::Float64)
	for ind in eachindex(lambda)
		for t in eachindex(lambda[ind])
			set_parameter_value(cache.lam[t, ind], lambda[ind][t])
			set_parameter_value(cache.prop[t, ind], proposed_coals[ind][t])
		end
	end
	# P3.2: Warm-start SCS from previous primal solution
	if has_values(cache.model)
		for v in all_variables(cache.model)
			set_start_value(v, value(v))
		end
	end
	optimize!(cache.model)
	return value.(cache.coal_exch)
end

function single_optimise_ADMM(opt::MPC_optimiser,bs::Vector{MPC_Building},k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	num_builds = bs.size[1]
	if num_builds == 1 || RESIL.enabled   # the coalition floor couples the members; it is only in single_optimise, not in ADMM yet
		return single_optimise(opt,bs,k,num_look_ahead,receding_horizon)[2], 1
	end
	if !receding_horizon
		num_steps = min(length(bs[1].act_cons)-k+1, num_look_ahead)
	else
		num_steps = num_look_ahead
	end
	num_steps = min(length(pred_consumption(bs[1],k)),num_steps)
	num_steps = max(num_steps, 1)
	c=0.5
	new_coal = zeros(num_steps,num_builds)
	proposed_coal =[zeros(num_steps) for i in 1:num_builds]
	lambdas = [zeros(num_steps) for i in 1:num_builds]
	num_iters_admm = 1
	# need a solution in case not converged (use new_coal)
	poss_sol = proposed_coal
	# P3.1: Build per-building and coal-update models ONCE; reuse across ADMM iterations.
	build_caches = [ADMM_build_opt_init(b, k, num_steps) for b in bs]
	coal_cache = ADMM_coal_update_init(num_builds, num_steps)
	# P3.7: Restored absolute tolerance 1e-5 (matching the original code that gave
	# the 489 benefit). The relative 1e-4 was ~100000x looser for large-magnitude
	# coalitions, giving less-converged solutions and different split-check outcomes.
	max_admm_iters = 1000
	abs_tol = 1e-5
	let states = Vector{Any}(undef,num_builds)
		while (norm(reduce(hcat, proposed_coal).-new_coal) > abs_tol) | (num_iters_admm <= 1)
			if num_iters_admm > max_admm_iters
				break
			end
			# c *= 1.1
		states = Vector{Any}(undef,num_builds)
		for ind in 1:num_builds
			states[ind] = ADMM_build_opt_solve(build_caches[ind], new_coal[:,ind], lambdas[ind], c)
		end
			proposed_coal = [value(state[6]) for state in states]

			new_coal = ADMM_coal_update_solve(coal_cache, lambdas, proposed_coal, c)
			if !any(isnan,new_coal)
				poss_sol = new_coal
			end

			lambdas = [lambda + c*(prop_coal-new_coal[:,ind]) for (lambda, prop_coal, ind) in zip(lambdas, proposed_coal, 1:num_builds)]
			num_iters_admm += 1
			# if num_iters_admm > 1000
			# 	println("Convergence failed")
			# 	break
			# end
			# println(proposed_coal)
			# z update
			#lambda update
		end
		# if num_iters_admm < 1000
		# 	println("Converged in limit")
		# end
		for state in states
			state[6] = poss_sol
		end
		states = [reduce(hcat,[state[i] for state in states]) for i in 1:length(states[1])]
		return states, num_iters_admm
	end	
end

function single_optimise_ADMM(opt::MPC_optimiser,b::MPC_Building,k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	return single_optimise_ADMM(opt,[b],k,num_look_ahead,receding_horizon)
end
function single_optimise(opt::MPC_optimiser,b::MPC_Building,k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	return single_optimise(opt,[b],k,num_look_ahead,receding_horizon)
end
function single_optimise_outage(opt::MPC_optimiser,b::MPC_Building,k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	return single_optimise_outage(opt,[b],k,num_look_ahead,receding_horizon)
end

function optimise(opt::MPC_optimiser,bs::Vector{MPC_Building},num_look_ahead::Int,ADMM::Bool=true,receding_horizon::Bool=false)
	num_steps = length(bs[1].act_cons)
	num_builds = length(bs)
	buy = zeros(num_steps,num_builds)
	sell = zeros(num_steps,num_builds)
	num_iters = 0
	for k = 1:num_steps
		# res = single_optimise_ADMM(opt, bs, k,true)
		# println(sum(value(res[5])))
		# model, res = single_optimise(opt, bs, k,true)
		# println(sum(value(res[5])))

		if ADMM
			res, num_iters_k = single_optimise_ADMM(opt, bs, k,num_look_ahead,receding_horizon)
			num_iters += num_iters_k
		else
			model, res = single_optimise(opt, bs, k,num_look_ahead,receding_horizon)
			num_iters += 1
		end
		for (i, b) in enumerate(bs)
			if k < num_steps
				b.SoC[k+1] = max(0,value(res[4][2,i])) #value(res[2][1,i]-res[3][1,i]-res[6][1,i])-b.act_cons[k]+b.act_prod[k]
				#println(b.SoC[k+1])
			end
			remaining = b.act_prod[k]-b.act_cons[k]-value(res[6][1,i]+res[1][1,i])
			if remaining > 0
				sell[k,i] = remaining
			else
				buy[k,i]=-remaining
			end		
		end
	end
	if num_steps <= 96
		buy_cost = opt.energy_cost[1:num_steps]
		sell_price = opt.energy_sale[1:num_steps]
	else
		buy_cost = hcat(repeat(opt.energy_cost,num_steps÷96), opt.energy_cost[1:(num_steps%96)])'
		sell_price = hcat(repeat(opt.energy_sale,num_steps÷96), opt.energy_sale[1:(num_steps%96)])'
	end
	cost = sum(buy_cost'*buy-sell_price'*sell)
	return cost, [buy, sell], num_iters/num_steps
	# return single_optimise(opt, bs, 1)
end

function single_coal_opt(coal_former::Function,bs::Vector{MPC_Building}, max_coal_size::Int, k::Int)
	coal = coal_former(bs,max_coal_size,k)
    # println(coal)
    # println(typeof(coal))
    if coal isa Vector{Vector{MPC_Building}}
        outs = [single_optimise(opt, agent, k) for agent in coal]
    else
        outs = [single_optimise(opt, buildings[agent], k) for agent in coal]
    end
    res = [sum(objective_value(out[1])) for out in outs]
    return coal, sum(res)
end

# function enumerate(itt::MPC_Building)
# 	return 1, itt
# end

function coal_stability_score(coal,old_coal)
	bs = reduce(vcat, coal)
	num_builds = length(bs)

	z_sum = 0
	for i in 1:num_builds
		for j in i+1:num_builds # wrong, needs to be from i to N
			together = false
			for agent in coal
				if bs[i] in agent && bs[j] in agent
					together = true
					break
				end
			end
			old_together = false
			for agent in old_coal
				if bs[i] in agent && bs[j] in agent
					if together 
						old_together = true
						break
					end
				end
			end
			z_sum += (together==old_together)
		end
	end
	return z_sum/binomial(num_builds,2)
end

function coal_MPC(coal_former::Function,bs::Vector{MPC_Building}, max_coal_size::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    num_steps = length(bs[1].act_cons)
	num_builds = length(bs)
	buy = zeros(num_steps,num_builds) #48*8 = 384 length
	sell = zeros(num_steps,num_builds)
	shed = zeros(num_steps,num_builds)   # outage steps: load that cannot be supplied
	curt = zeros(num_steps,num_builds)   # outage steps: generation curtailed
	num_iters = 0
	old_coal = 0
	stab_score = []
	coal_hist = Any[]   # coalition structure chosen at each step
	for k = 1:num_steps
        coal, outs, num_iters_k = coal_former(bs,max_coal_size,k,num_look_ahead,receding_horizon)
		push!(coal_hist, deepcopy(coal))
		num_iters += num_iters_k
        # if coal isa Vector{Vector{MPC_Building}}
        #     outs = [single_optimise_ADMM(opt, agent, k) for agent in coal]
        # else
        #     outs = [single_optimise_ADMM(opt, buildings[agent], k) for agent in coal]
        # end
        #res = [sum(value(out[5])) for out in outs]
		# _, res = single_optimise(opt, bs, k)
		if k > 1
			curr_stability_score =coal_stability_score(coal, old_coal)
			append!(stab_score,curr_stability_score)
			# println(stability_score)
		end
		old_coal = coal
		for (res,agent) in zip(outs, coal)
            if !(coal isa Vector{Vector{MPC_Building}})
                agent = bs[agent]
            end
			if !(agent isa MPC_Building)
				for (i,b) in enumerate(agent)
					if k < num_steps
						# P3.3: guard against NaN from a non-converged SCS solve
						next_soc = value(res[4][2,i])
						b.SoC[k+1] = isnan(next_soc) ? b.SoC[k] : max(0, next_soc)
					end
					# println(b.SoC)
					remaining = b.act_prod[k]-b.act_cons[k]-(value(res[6][1,i]) + value(res[1][1,i]))
					isnan(remaining) && (remaining = 0.0)
					if in_outage(k)   # no grid: a surplus is curtailed, a deficit is load that cannot be supplied
						remaining > 0 ? (curt[k,b.id] = remaining) : (shed[k,b.id] = -remaining)
					elseif remaining > 0
						sell[k,b.id] = remaining
					else
						buy[k,b.id]=-remaining
					end
					# println(remaining)
				end
			else
				b = agent
				if k < num_steps
					next_soc = value(res[4][2,1])
					b.SoC[k+1] = isnan(next_soc) ? b.SoC[k] : max(0, next_soc)
				end
				remaining = b.act_prod[k]-b.act_cons[k]-(value(res[6][1,1]) + value(res[1][1,1]))
				isnan(remaining) && (remaining = 0.0)
				if in_outage(k)   # no grid: a surplus is curtailed, a deficit is load that cannot be supplied
					remaining > 0 ? (curt[k,b.id] = remaining) : (shed[k,b.id] = -remaining)
				elseif remaining > 0
					sell[k,b.id] = remaining
				else
					buy[k,b.id]=-remaining
				end
			end
		end
	end
	if num_steps <= 96
		buy_cost = opt.energy_cost[1:num_steps]
		sell_price = opt.energy_sale[1:num_steps]
	else
		buy_cost = hcat(repeat(opt.energy_cost,num_steps÷96), opt.energy_cost[1:(num_steps%96)])'
		sell_price = hcat(repeat(opt.energy_sale,num_steps÷96), opt.energy_sale[1:(num_steps%96)])'
	end
	cost = sum(buy_cost'*buy-sell_price'*sell)
	return cost, [buy, sell], num_iters/num_steps, stab_score, coal_hist, [shed, curt]
end
