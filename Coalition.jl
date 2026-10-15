include("MPC_optimiser.jl")
using Combinatorics
using StatsBase
using Base.Threads

# =============================================================================
# Coalition formation — two paradigms (see AGENTS.md):
#
#   Limited information (non-DP) — privacy_focussed_coals
#     Agents share only their net consumption vectors, not full cost functions.
#     Deterministic greedy merge by compatibility value + myopic split-check.
#     No differential-privacy guarantee. Fully reproducible.
#
#   Differential privacy (DP) — privacy_focussed_coals_with_delta
#     Exponential mechanism over pair values; delta_G is the privacy budget
#     (delta_G=0 -> argmax/deterministic pick; larger -> softer, more private
#     sampling). All pairs enter the pool (incompatible ones get value 0) so the
#     candidate set does not leak data. The myopic split-check is removed
#     (it would break the DP guarantee). Stochastic — set a seed.
# =============================================================================

# Return the building-id list for an agent (Int singleton or Vector coalition)
agent_ids(a::Int) = [a]
agent_ids(a::Vector) = a

# ---- Switches for the limited-information formation, items 1-3 of code_vs_paper_discrepancies.txt ----
# The defaults reproduce the code the paper's results came from.
Base.@kwdef mutable struct FormationParams
    order::Symbol = :paper       # :paper     visit pairs from the SMALLEST val up, no val > 0 test (item 2)
                                 # :val_desc  Algorithm 1 as written: largest val first, merge only if val > 0
    exact_bound::Bool = false    # true: eq. (18) with an element-wise min (min.); false: the code's vector min (item 1)
    fix_dissolve::Bool = false   # true: apply the (22) splits if ANY coalition fails; false: only if the LAST agent passed (item 3)
end
const FORM = FormationParams()
const VAL_TOL = 1e-6   # merge values below this count as zero (solver noise around Q = 0)

# DIAGNOSTIC: global log for coalition pick instrumentation. Each entry:
# (k, round, picked_ids, picked_val, argmax_ids, argmax_val, is_argmax, delta_G)
# Cleared by the caller before each run; read after to analyse picks.
# SECURITY: the log records the full sampling transcript (which pairs were
# considered and their utilities) — it MUST NOT be exposed in a deployment
# where the DP guarantee on the coalition structure is relied upon. Gate the
# push! with LOG_PICKS so the transcript is only recorded when debugging.
const LOG_PICKS = false
global PICK_LOG = Tuple{Int,Int,Vector{Int},Float64,Vector{Int},Float64,Bool,Float64}[]


function find_opt_coal(buildings::Vector{Building}, max_coal_size::Int)
    possible_coals = collect(partitions(buildings))
    coal_vals = Dict()
    for coal in possible_coals
	    valid_coal = true
	    for elem in coal
		    if elem.size[1] > max_coal_size
			    valid_coal = false
			    break
		    end
	    end
	    if valid_coal
            obj_val=0
            for ind_coal in coal
                coal_res = optimise(opt,ind_coal)
                obj_val += sum(value(coal_res[2][5]))
            end
            coal_vals[coal] = obj_val
	    end
    end
    sorted_coal_vals = sort!(collect(coal_vals), by=last)
    return sorted_coal_vals[1][1], sorted_coal_vals[1][2], length(possible_coals)
end

function bottom_up_full_info(buildings::Vector{Building}, max_coal_size::Int)
    agents = Vector(1:length(buildings))
    done = false
    outs = [optimise(opt, buildings[agent]) for agent in agents]
    num_iters = 1

    coal_vars = Dict()
    coal_vals = Dict()
    for (out, agent) in zip(outs, agents)
        coal_vars[agent] = out[1]
        coal_vals[[agent]] = sum(value(out[2][5]))
    end
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        outs = [optimise(opt, buildings[agent]) for agent in agents]
        res = Dict([agent => out[1] for (out, agent) in zip(outs, agents)])
        vars = [out[2] for out in outs]

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        poss_coal_vars = Dict()
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            coal_res = optimise(opt, buildings[poss_coal_vec])
            obj_val = sum(value(coal_res[2][5]))
            poss_coal_vars[c] = coal_res[1]
            if c[1] isa Int64
                c_1 = [c[1]]
            else
                c_1 = c[1]
            end
            if c[2] isa Int64
                c_2 = [c[2]]
            else
                c_2 = c[2]
            end
            poss_coal_vals[c] = -((coal_vals[c_1])+(coal_vals[c_2]))+ obj_val
            # poss_coal_vals[c] = -(objective_value(res[c[1]])+objective_value( res[c[2]]))+ sum(objective_value(optimise(opt, buildings[poss_coal_vec])[1])) #added res c[1] c[2] - joint to better discriminate
        end
        sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        new_agents = Vector()
        coaled_agents = Vector()
        for elem in sorted_coal_vals
            if !(elem[1][1] in coaled_agents) && !(elem[1][2] in coaled_agents)
                new_coal = Vector()
                append!(new_coal, elem[1][1], elem[1][2])
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    push!(coaled_agents, elem[1][1], elem[1][2])
                    coal_vals[new_coal] = poss_coal_vals[elem[1]]
                    coal_vars[new_coal] = poss_coal_vars[elem[1]]
                    done = false
                end
            end
        end
        for agent in agents
            if !(agent in coaled_agents)
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        #println(sorted_coal_vals)
        #update cons_vec
    end

    return agents, [coal_vars[agent] for agent in agents], num_iters

end

function privacy_focussed_coals(buildings::Vector{Building}, max_coal_size::Int)
    agents = Vector(1:length(buildings))
    done = false
    energy_diff = opt.energy_cost-opt.energy_sale
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        outs = [optimise(opt, buildings[agent]) for agent in agents]
        res = [out[1] for out in outs]
        vars = [out[2] for out in outs]

        cons_vec = Dict([agent=> vec(sum(value(var[2]-var[3]),dims=2)) for (agent, var) in zip(agents,vars)])

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            compatible_slots = cons_vec[c[1]].*cons_vec[c[2]] .< zeros(length(cons_vec[c[1]]))
            if any(compatible_slots)
                poss_coal_vals[c] = energy_diff*(min(abs.(compatible_slots.*cons_vec[c[1]]),abs.(compatible_slots.*cons_vec[c[2]]))) # should be dotted with price vector
                #poss_coal_vals[c] = norm(cons_vec[c[1]]+cons_vec[c[2]])
            end
        end
        # sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        new_agents = Vector()
        coaled_agents = Vector()
        done = false
        delta_u = 1 # change this!
        println("Needs edit")
        epsilon = 1e-1
        weights = [Float64(exp(epsilon*i[2]/(2*delta_u))) for i in sorted_coal_vals]
        poss_coals_for_samples = [elem[1] for elem in sorted_coal_vals]
        while not done
            elem = sample(poss_coal_vals, Weights(weights))
            if !(elem[1] in coaled_agents) && !(elem[2] in coaled_agents)
                new_coal = Vector()
                append!(new_coal, elem[1], elem[2])
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    push!(coaled_agents, elem[1], elem[2])
                end
            end
            ind = findfirst(==(elem), poss_coal_vals)
            poss_coals_for_samples = vcat(poss_coals_for_samples[1:ind-1], poss_coals_for_samples[ind+1:length(poss_coals_for_samples)])
            weights = vcat(weights[1:ind-1], weights[ind+1:length(weights)])
            if length(poss_coals_for_samples) == 0
                done = true
            end
        end
        # for elem in sorted_coal_vals
        #     if !(elem[1][1] in coaled_agents) && !(elem[1][2] in coaled_agents)
        #         new_coal = Vector()
        #         append!(new_coal, elem[1][1], elem[1][2])
        #         if length(new_coal) <= max_coal_size
        #             push!(new_agents, new_coal)
        #             push!(coaled_agents, elem[1][1], elem[1][2])
        #             done = false
        #         end
        #     end
        # end
        for agent in agents
            if !(agent in coaled_agents)
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        #println(sorted_coal_vals)
        #update cons_vec
    end
    #println(cons_vec)
    #println("cons ", value(vars[2][2]))
    # #println("sell ", vars[1][3])
    # outs = [optimise(opt, buildings[agent]) for agent in agents]
    # res = [sum(objective_value(out[1])) for out in outs]
    # println(agents)
    outs = [optimise(opt, buildings[agent]) for agent in agents]
    res = [out[1] for out in outs]
    vars = [sum(value(out[2][5])) for out in outs]
    return agents, vars, 1
end

function find_opt_coal(buildings::Vector{MPC_Building}, max_coal_size::Int,k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    possible_coals = collect(partitions(buildings))
    coal_vals = Dict()
    num_iters = 0
    for coal in possible_coals
	    valid_coal = true
	    for elem in coal
		    if elem.size[1] > max_coal_size
			    valid_coal = false
			    break
		    end
	    end
	    if valid_coal
            obj_val = 0
            for ind_coal in coal
                coal_res = single_optimise_ADMM(opt,ind_coal,k,num_look_ahead,receding_horizon)
                num_iters += coal_res[2]
                obj_val += sum(value(coal_res[1][5]))
            end
            coal_vals[coal] = obj_val
	    end
    end
    sorted_coal_vals = sort!(collect(coal_vals), by=last)
    outs = [single_optimise_ADMM(opt, agent, k)[1] for agent in sorted_coal_vals[1][1]] #can do this in the loop, so don't count these iters
    return sorted_coal_vals[1][1], outs, num_iters#, sorted_coal_vals[1][2]
end

function bottom_up_full_info(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    agents = Vector(1:length(buildings))
    done = false
    outs = [single_optimise_ADMM(opt, buildings[agent], k,num_look_ahead,receding_horizon) for agent in agents]
    num_iters = sum([out[2] for out in outs])

    coal_vars = Dict()
    coal_vals = Dict()
    for (out, agent) in zip(outs, agents)
        coal_vars[agent] = out[1]
        coal_vals[[agent]] = sum(value(out[1][5]))
    end
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        poss_coal_vars = Dict()
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            coal_res = single_optimise_ADMM(opt, buildings[poss_coal_vec], k,num_look_ahead,receding_horizon)
            num_iters += coal_res[2]
            obj_val = sum(value(coal_res[1][5]))
            poss_coal_vars[c] = coal_res[1]
            if c[1] isa Int64
                c_1 = [c[1]]
            else
                c_1 = c[1]
            end
            if c[2] isa Int64
                c_2 = [c[2]]
            else
                c_2 = c[2]
            end
            poss_coal_vals[c] = -((coal_vals[c_1])+(coal_vals[c_2]))+ obj_val #added poss_coal_vals c[1] c[2] - joint to better discriminate
            # poss_coal_vals[c] = -(objective_value(res[c[1]])+objective_value( res[c[2]]))+ sum(obj_val) #added res c[1] c[2] - joint to better discriminate
        end
        sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        new_agents = Vector()
        coaled_agents = Vector()
        for elem in sorted_coal_vals
            if !(elem[1][1] in coaled_agents) && !(elem[1][2] in coaled_agents)
                new_coal = Vector()
                append!(new_coal, elem[1][1], elem[1][2])
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    push!(coaled_agents, elem[1][1], elem[1][2])
                    coal_vals[new_coal] = poss_coal_vals[elem[1]]
                    coal_vars[new_coal] = poss_coal_vars[elem[1]]
                    done = false
                end
            end
        end
        for agent in agents
            if !(agent in coaled_agents)
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        #println(sorted_coal_vals)
        #update cons_vec
    end
    #println(cons_vec)
    #println("cons ", value(vars[2][2]))
    #println("sell ", vars[1][3])
    # outs = [single_optimise(opt, buildings[agent], k) for agent in agents]
    # res = [sum(objective_value(out[1])) for out in outs]
    return agents, [coal_vars[agent] for agent in agents], num_iters
end

"""
    privacy_focussed_coals_with_delta(buildings, max_coal_size, k, num_look_ahead, receding_horizon=false, delta_G=100.0)

Differentially-private coalition formation for MPC buildings (exponential mechanism).

`delta_G` is the privacy budget: 0.0 → argmax (deterministic pick, still
mechanism-shaped but no randomness), larger → softer sampling (more private).
All candidate pairs enter the sampling pool (incompatible pairs get value 0)
so the candidate set does not depend on the data. No post-hoc split-check (it
would violate the DP guarantee). Stochastic — call `Random.seed!` for
reproducibility.

Returns `(agents, vars, num_iters)`.
"""
function privacy_focussed_coals_with_delta(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int,num_look_ahead::Int,receding_horizon::Bool=false, delta_G::Float64=100.0)
    agents = Vector(1:length(buildings))
    done = false
    energy_diff = opt.energy_cost-opt.energy_sale
    # Compute correct delta_u bound: max timestep difference (buy price - sell price) * delta_G
    max_price_diff = maximum(abs.(opt.energy_cost - opt.energy_sale))
    # Add minimum threshold to prevent numerical issues with very small delta_G
    delta_u = max(delta_G * max_price_diff, 1e-10)
    num_iters=0
    vars = 0
    # P1.2: Cache singleton results: building_id => (vars, cons_vec)
    singleton_cache = Dict{Int, Tuple{Any, Vector{Float64}}}()
    cache_lock = SpinLock()
    round_num = 0
    while !done
        round_num += 1
        done = true
        # P1.2 + P1.5: Cache singleton solves across rounds; parallelise across agents.
        outs = Vector{Any}(undef, length(agents))
        @threads for i in 1:length(agents)
            agent = agents[i]
            if agent isa Int
                # Singleton: check cache (with lock for read), solve + cache if missing
                local cv
                cached = false
                lock(cache_lock) do
                    cached = haskey(singleton_cache, agent)
                    if cached
                        outs[i] = (singleton_cache[agent][1], 0)
                    end
                end
                if !cached
                    res = single_optimise_ADMM(opt, buildings[agent], k, num_look_ahead, receding_horizon)
                    cv = vec(sum(value(res[1][2]-res[1][3]), dims=2))
                    lock(cache_lock) do
                        singleton_cache[agent] = (res[1], cv)
                    end
                    outs[i] = (res[1], res[2])
                end
            else
                res = single_optimise_ADMM(opt, buildings[[agent...]], k, num_look_ahead, receding_horizon)
                outs[i] = (res[1], res[2])
            end
        end
        vars = [out[1] for out in outs]
        num_iters += sum([out[2] for out in outs])
        #res = [out[1] for out in outs]

        cons_vec = Dict{Any, Vector{Float64}}()
        for (agent, var) in zip(agents, vars)
            if agent isa Int && haskey(singleton_cache, agent)
                cons_vec[agent] = singleton_cache[agent][2]
            else
                # P3.4: NaN-guard — replace NaN from non-converged SCS with 0.0 so a
                # bad coalition solve doesn't poison the next round's poss_coal_vals/softmax.
                cv = vec(sum(value(var[2]-var[3]), dims=2))
                cons_vec[agent] = [isnan(x) ? 0.0 : x for x in cv]
            end
        end

        # P1.4: Vectorise the O(n^2) pair compatibility check.
        # Build cons_mat (horizon × n_agent) and compute all pair values with matrix ops.
        n_agents = length(agents)
        horizon = min(length(buildings[1].act_cons)-k+1, num_look_ahead)
        cons_mat = reduce(hcat, [cons_vec[a][1:horizon] for a in agents])  # horizon × n_agent
        abs_mat = abs.(cons_mat)  # horizon × n_agent
        # sign_compat[t,i,j] = cons_mat[t,i]*cons_mat[t,j] < 0  (opposite signs => can trade)
        # min_pair[t,i,j]    = min(abs_mat[t,i], abs_mat[t,j])
        # Use 3D broadcasting: abs_mat (h×n×1) vs abs_mat' reshaped (1×n×h)? Easier: explicit loops over t (horizon is small, ~2-8).
        diff_vec = energy_diff[1:horizon]
        pair_val_mat = zeros(n_agents, n_agents)
        any_compat = falses(n_agents, n_agents)
        for t in 1:horizon
            row = cons_mat[t, :]            # n_agent
            opp = (row .* row') .< 0        # n_agent × n_agent: signs differ
            m = min.(abs.(row), abs.(row')) # n_agent × n_agent: pairwise min of magnitudes
            pair_val_mat .+= diff_vec[t] .* (opp .* m)
            any_compat .|= opp
        end
        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        # DP: every pair enters the pool unconditionally. pair_val_mat[i,j] is
        # already 0 for incompatible pairs (the opp.*m term vanishes), so they
        # get value 0 and a correspondingly small (but positive) softmax weight
        # and remain samplable. Filtering them out would leak the compatibility
        # structure of the data through the sampler's candidate set.
        for c in poss_coals
            i = findfirst(==(c[1]), agents)
            j = findfirst(==(c[2]), agents)
            poss_coal_vals[c] = pair_val_mat[i, j]
        end
        # sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        #
        new_agents = Vector()
        # P2.1: Track whether any merge was actually accepted this round; if not, exit.
        merge_happened = false
        epsilon = 1e-0
        ks = collect(keys(poss_coal_vals))
        wv = [epsilon*poss_coal_vals[i]/(2*delta_u) for i in ks]
        # Numerically stable softmax: subtract max before exp() to prevent
        # overflow without flattening large weights into a uniform distribution.
        mw = isempty(wv) ? 0.0 : maximum(wv)
        weights = [exp(w - mw) for w in wv]
        # Ensure weights are finite and valid
        weights = [isfinite(w) ? w : 1.0 for w in weights]
        n_pool = length(ks)
        # Phase A: instead of a coaled_agents membership check on every draw,
        # when a merge is accepted we eagerly remove ALL pairs overlapping
        # with the new coalition from the pool. This shrinks the pool faster
        # (so each subsequent sample(1:n_pool, Weights(...)) builds a smaller
        # alias table) and eliminates the O(|coaled|) membership check.
        while n_pool > 0
            try
                # delta_G=0 -> argmax (deterministic; avoids sample() rounding noise
                # from SCS-inflated weights at delta_u=1e-10). delta_G>0 -> softmax
                # sample. argmax still depends on SCS noise in poss_coal_vals but
                # is reproducible per machine (no alias-table RNG consumed).
                ind = delta_G <= 0.0 ? argmax(weights[1:n_pool]) : sample(1:n_pool, Weights(weights[1:n_pool]))
                elem = ks[ind]
                # DIAGNOSTIC: log the pick vs argmax (before removal shuffles the pool).
                # picked_ids/argmax_ids are the flat building-id lists of each pair.
                if LOG_PICKS
                    am_ind = argmax(weights[1:n_pool])
                    pk_ids = sort(vcat(agent_ids(elem[1]), agent_ids(elem[2])))
                    am_ids = sort(vcat(agent_ids(ks[am_ind][1]), agent_ids(ks[am_ind][2])))
                    push!(PICK_LOG, (k, round_num, pk_ids, poss_coal_vals[elem],
                                     am_ids, poss_coal_vals[ks[am_ind]],
                                     ind == am_ind, delta_G))
                end
                # Remove the drawn pair (swap-with-last, O(1))
                weights[ind] = weights[n_pool]
                ks[ind] = ks[n_pool]
                n_pool -= 1
                # Form the candidate coalition as a flat id list
                new_coal = vcat(agent_ids(elem[1]), agent_ids(elem[2]))
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    merge_happened = true
                    # Eagerly remove every pair in the pool that overlaps with new_coal.
                    # Each pair is removed at most once per round -> O(n^2) total.
                    coaled_ids = Set(new_coal)
                    i = 1
                    while i <= n_pool
                        pair_ids = vcat(agent_ids(ks[i][1]), agent_ids(ks[i][2]))
                        if !isempty(intersect(pair_ids, coaled_ids))
                            weights[i] = weights[n_pool]
                            ks[i] = ks[n_pool]
                            n_pool -= 1
                        else
                            i += 1
                        end
                    end
                end
            catch e
                # In case of weight problems, fall back to uniform sampling
                println(e)
                if length(ks) > 0
                    # If we're in this catch block, there might be issues with weights
                    # Fall back to choosing a random pair
                    ind = rand(1:min(n_pool, length(ks)))
                    elem = ks[ind]
                    # Remove the drawn pair (swap-with-last, O(1))
                    weights[ind] = weights[n_pool]
                    ks[ind] = ks[n_pool]
                    n_pool -= 1
                    # Form the candidate coalition as a flat id list
                    new_coal = vcat(agent_ids(elem[1]), agent_ids(elem[2]))
                    if length(new_coal) <= max_coal_size
                        push!(new_agents, new_coal)
                        merge_happened = true
                    end
                else
                    break
                end
            end
        end
        # Phase A.3: derive coaled_ids from the accepted coalitions in new_agents
        coaled_ids = Set{Int}()
        for na in new_agents
            union!(coaled_ids, agent_ids(na))
        end
        for agent in agents
            if !any(id in coaled_ids for id in agent_ids(agent))
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        # P2.1: Exit when no merge was accepted this round (instead of running until poss_coal_vals is empty)
        done = !merge_happened
        #println(sorted_coal_vals)
        #update cons_vec
    end
    # P3.8: Removed the myopic split check. It negated DP guarantees by setting
    # coalitions to zero probability in a data-dependent way (dec_val and
    # coal_single_val both depend on the data, so the keep/split decision is
    # not covered by the post-processing theorem). Coalitions formed by the
    # exponential mechanism are now kept as-is. The UB (poss_coal_vals) is
    # still the utility, so bad coalitions get low sampling probability
    # naturally.
    # P1.3: Reuse existing vars (no splits to re-solve).
    vars_by_agent = Dict{Any, Any}(zip(agents, vars))
    vars = Vector{Any}(undef, length(agents))
    for (i, agent) in enumerate(agents)
        if haskey(vars_by_agent, agent)
            vars[i] = vars_by_agent[agent]
        elseif agent isa Int && haskey(singleton_cache, agent)
            vars[i] = singleton_cache[agent][1]
        else
            res = single_optimise_ADMM(opt, buildings[agent], k, num_look_ahead, receding_horizon)
            num_iters += res[2]
            cv = vec(sum(value(res[1][2]-res[1][3]), dims=2))
            singleton_cache[agent] = (res[1], cv)
            vars[i] = res[1]
        end
    end
    return agents, vars, num_iters
end

"""
    privacy_focussed_coals(buildings, max_coal_size, k, num_look_ahead, receding_horizon=false)

Limited-information (NON-DP) coalition formation for MPC buildings.

Agents share only their net consumption vectors (not full cost functions).
Deterministically forms coalitions by: (1) solving each agent/coalition via
ADMM, (2) scoring pairs by energy-trading compatibility, (3) greedily merging
the best non-overlapping pairs up to `max_coal_size`, (4) a myopic split-check
that dissolves any coalition whose joint first-step cost exceeds the members
acting alone.

Returns `(agents, vars, num_iters)`. Deterministic — no RNG.
"""
function privacy_focussed_coals(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    FORM.order in (:paper, :val_desc) || error("FORM.order must be :paper or :val_desc")
    agents = Vector(1:length(buildings))           #A_new ← 所有 building
    done = false
    energy_diff = opt.energy_cost-opt.energy_sale
    num_iters=0
    vars = 0
    dec_single_vals = 0
    singleton_cache = Dict{Int, Tuple{Any, Vector{Float64}}}()
    first_round = true
    while !done
        done = true
        outs = Vector{Any}(undef, length(agents))
        for i in 1:length(agents)                   #每个 agent 解问题 (9), agent is a number list
            agent = agents[i]
            if agent isa Int && haskey(singleton_cache, agent)
                outs[i] = (singleton_cache[agent][1], 0)
            elseif agent isa Int
                res = single_optimise_ADMM(opt, buildings[agent], k, num_look_ahead, receding_horizon)
                cv = vec(sum(value(res[1][2]-res[1][3]), dims=2)) #difference between buy/sell，if the value is negative, it means the agent is selling energy, if positive, it means the agent is buying energy in total
                cv = [isnan(x) ? 0.0 : x for x in cv] 
                singleton_cache[agent] = (res[1], cv)
                outs[i] = (res[1], res[2])
            else
                res = single_optimise_ADMM(opt, buildings[[agent...]], k, num_look_ahead, receding_horizon)
                outs[i] = (res[1], res[2])
            end
        end
        vars = [out[1] for out in outs]
        if first_round
            dec_single_vals = [value(out[2][1])*energy_cost_k(opt,k,1)-value(out[3][1])*energy_sale_k(opt,k,1) for out in vars]
            first_round = false
        end
        num_iters += sum([out[2] for out in outs])

        cons_vec = Dict{Any, Vector{Float64}}()       #算 val, is this only the net energy exchange?
        for (agent, var) in zip(agents, vars)
            if agent isa Int && haskey(singleton_cache, agent)
                cons_vec[agent] = singleton_cache[agent][2]
            else
                cv = vec(sum(value(var[2]-var[3]), dims=2))
                cons_vec[agent] = [isnan(x) ? 0.0 : x for x in cv]
            end
        end

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            num_look_ahead = min(length(buildings[1].act_cons)-k+1, num_look_ahead)
            compatible_slots = (cons_vec[c[1]].*cons_vec[c[2]] .< zeros(length(cons_vec[c[1]])))[1:num_look_ahead] # 1 or 0 in the compatible slots based
            if any(compatible_slots)
                qa = abs.(compatible_slots.*cons_vec[c[1]]) 
                qb = abs.(compatible_slots.*cons_vec[c[2]])
                # item 1: eq. (18) needs the element-wise min.; the code's min compares whole vectors
                # was min( abs.(compatible_slots.*cons_vec[c[1]]), abs.(compatible_slots.*cons_vec[c[2]]) ) # should be dotted with price vector
                # item 4: energy_diff[1:H] prices steps 1..H, but cons_vec covers steps k..k+H-1; eq. (18) needs energy_cost_k/energy_sale_k at k
                poss_coal_vals[c] = energy_diff[1:num_look_ahead]'*(FORM.exact_bound ? min.(qa, qb) : min(qa, qb)) ##
            end
        end
        # item 2: :paper visits pairs from the smallest val up; :val_desc from the largest down
        sorted_coal_vals = sort!(collect(poss_coal_vals), by=last, rev = FORM.order == :val_desc)  #sort ##
        new_agents = Vector()
        coaled_agents = Vector()
        for elem in sorted_coal_vals     # 按顺序接受互不重叠的 pair
            FORM.order == :val_desc && elem[2] <= VAL_TOL && continue   # Algorithm 1: merge only if val > 0
            if !(elem[1][1] in coaled_agents) && !(elem[1][2] in coaled_agents)
                new_coal = Vector()
                append!(new_coal, elem[1][1], elem[1][2])
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    push!(coaled_agents, elem[1][1], elem[1][2])
                    done = false
                end
            end
        end
        for agent in agents   #加回没配对的 agent
            if !(agent in coaled_agents)
                push!(new_agents, agent)
            end
        end

        agents = new_agents
    end
    pre_split_agents = agents   # store the agents before the check
    pre_split_vars = vars   # the vars contain the results of the optimisation for each agent
    new_agents = Vector()
    added = false
    any_split = false
    for (agent, var) in zip(agents,vars)
        added = false
        if length(agent) > 1
            dec_val = sum(sum(dec_single_vals[i] for i in agent))
            coal_single_val = sum(value(var[2][1,:]).*energy_cost_k(opt,k,1)-value(var[3][1,:]).*energy_sale_k(opt,k,1))
            if dec_val >= coal_single_val
                push!(new_agents, agent)
                added = true
            else
                for i in agent
                    push!(new_agents, i)
                end
                any_split = true
            end
        else
            push!(new_agents,agent)
        end
    end
    # item 3: the code applies the splits only if the LAST agent was a coalition that passed (22)
    if FORM.fix_dissolve ? any_split : added
        agents = new_agents
    end
    vars_by_agent = Dict{Any, Any}(zip(pre_split_agents, pre_split_vars))
    vars = Vector{Any}(undef, length(agents))
    for (i, agent) in enumerate(agents)
        if haskey(vars_by_agent, agent)
            vars[i] = vars_by_agent[agent]
        elseif agent isa Int && haskey(singleton_cache, agent)
            vars[i] = singleton_cache[agent][1]
        else
            res = single_optimise_ADMM(opt, buildings[agent], k, num_look_ahead, receding_horizon)
            num_iters += res[2]
            vars[i] = res[1]
        end
    end
    return agents, vars, num_iters
end

const RES_TOL = 1e-3   # reserve margins that differ by less than this count as equal

# Energy exchanged inside a coalition in the first step of its plan: the sum of the members' positive coal_exch, i.e. the
# energy they send to the other members, which equals the energy the others receive because sum(coal_exch) = 0 (coal_c).
# No prices. A single building has no partner, so its value is 0. NaN if the solve failed.
function first_step_compensation(var, n::Int)
    n == 1 && return 0.0
    c = [value(var[6][1, i]) for i in 1:n]
    return any(isnan, c) ? NaN : sum(max(x, 0.0) for x in c)
end

# V(C) of a plan: the members' planned cost over the horizon; a failed solve counts as infinite cost
plan_cost(var) = (v = sum(value.(var[5])); isnan(v) ? Inf : v)

# Reserve margin of a coalition at step k, the resilience indicator of phase 1: the members' predicted PV plus stored
# energy minus predicted load at step k, minus the coalition's reserve s~_C for the next tau steps (stilde_coal, entry 1).
# > 0: energy left beyond the reserve; < 0: short of it. Data only, no plan needed.
function reserve_margin(bs::Vector{MPC_Building}, k::Int)
    net = sum(b.pred_prod[k, 1] + b.SoC[k] - b.pred_cons[k, 1] for b in bs)
    return net - stilde_coal(bs, k, 1, RESIL.tau)[1]
end

# Reachable margin of a coalition (method B): at each horizon step j = 2..H, the most total energy its batteries can hold
# when charging at full power from now on, minus the reserve s~_C required there (resil_floor); the worst step of the
# window. With the grid, charging at full power is always possible, so the coalition's problem is feasible if and only
# if this is >= 0. Data only, no plan needed. Inf when there is no reserve constraint.
function reach_margin(bs::Vector{MPC_Building}, k::Int, H::Int)
    (!RESIL.enabled || H < 2) && return Inf
    st = stilde_coal(bs, k, H, RESIL.tau)          # st[j]: reserve at horizon step j (j = 1 is step k)
    return minimum(sum(min(Float64(b.max_storage), b.SoC[k] + Float64(b.charge_eff)*Float64(b.storage_max_flow)*(j - 1))
                       for b in bs) - st[j] for j in 2:H)
end

# Resilience value used to rank coalitions: the worse of the reserve margin now and the reachable margin over the window
resilience_value(bs::Vector{MPC_Building}, k::Int, H::Int) = min(reserve_margin(bs, k), reach_margin(bs, k, H))

"""
    resilience_focussed_coals(buildings, max_coal_size, k, num_look_ahead, receding_horizon=false)

Coalition formation in two greedy bottom-up phases, each like Algorithm 1 of the paper. The resilience of an agent C is m(C) =
resilience_value, the worse of its reserve margin now (reserve_margin) and its reachable margin over the window
(reach_margin). 
At an outage step (in_outage(k)) the coalitions are formed again with the outage problem single_optimise_outage.
Returns `(agents, vars, num_iters)`.
"""
function resilience_focussed_coals(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    num_iters = 0
    p_buy, p_sell = energy_cost_k(opt, k, 1)[1], energy_sale_k(opt, k, 1)[1]
    first_step_cost(var) = sum(value.(var[2][1, :]))*p_buy - sum(value.(var[3][1, :]))*p_sell
    plans = Dict{Any, Any}()                       # agent => (plan, resilience value, total cost)
    merged = Dict{Vector{Int}, Any}()              # sorted member ids => (ids, plan, resilience value, total cost) of a solved merge
    H = receding_horizon ? num_look_ahead : min(length(buildings[1].act_cons) - k + 1, num_look_ahead)
    H = max(min(length(pred_consumption(buildings[1], k)), H), 1)   # horizon length, as in single_optimise
    # Outage step: no grid, so every plan is the outage problem (shed + curtailment cost) and the coalitions are formed
    # again at every step; there is no grid to charge from, so the reachable margin does not apply.
    outage = in_outage(k)
    solve_plan(ids) = outage ? (single_optimise_outage(opt, buildings[ids], k, num_look_ahead, receding_horizon)[2], 1) :
                               single_optimise_ADMM(opt, buildings[ids], k, num_look_ahead, receding_horizon)
    value_of(bs) = outage ? reserve_margin(bs, k) : resilience_value(bs, k, H)

    agents = Vector{Any}(1:length(buildings))      # A_new ← 所有 building
    for a in agents                                # 每个 building 单独解问题 (9)
        res = solve_plan([a])
        num_iters += res[2]
        plans[a] = (res[1], value_of(buildings[[a]]), plan_cost(res[1]))
    end
    # first-step cost of each building alone, for (22); an outage has no grid cost
    dec_single_vals = outage ? Float64[] : [first_step_cost(plans[a][1]) for a in agents]

    for phase in (:resilience, :cost)
        while true                                 # one round of pairwise merges
            candidates = []
            for (a, b) in combinations(agents, 2)
                ids = vcat(agent_ids(a), agent_ids(b))
                length(ids) <= max_coal_size || continue
                key = sort(ids)
                !outage && reach_margin(buildings[ids], k, H) < -RES_TOL && continue   # method B: no feasible plan, not solved
                if !haskey(merged, key)            # solve each merge once; phase 2 re-uses the solves of phase 1
                    res = solve_plan(ids)
                    num_iters += res[2]
                    merged[key] = (ids, res[1], value_of(buildings[ids]), plan_cost(res[1]))
                end
                ids, plan, m_ab, V = merged[key]
                isinf(V) && continue               # the merged problem has no solution
                lo = min(plans[a][2], plans[b][2])           # margin of the weaker of the two
                saving = plans[a][3] + plans[b][3] - V       # cost saved by merging

                if phase == :resilience
                    m_ab > lo + RES_TOL || continue                        # phase 1: only merges that lift the weaker one
                    score = (-lo, m_ab)                                    # weakest agent first, then its best partner
                else
                    (saving > VAL_TOL && m_ab >= lo - RES_TOL) || continue # phase 2: cheaper, weaker margin not lowered
                    score = saving
                end

                push!(candidates, (ids = ids, plan = plan, margin = m_ab, V = V, score = score))
            end
            
            sort!(candidates, by = c -> c.score, rev = true)  # best pair first (phase 1 scores are tuples)
            new_agents = Vector{Any}()
            taken = Set{Int}()
            for c in candidates                    # 按顺序接受互不重叠的 pair
                any(i in taken for i in c.ids) && continue
                push!(new_agents, c.ids)
                union!(taken, c.ids)
                plans[c.ids] = (c.plan, c.margin, c.V)
            end

            isempty(new_agents) && break           # no merge in this round: the phase is over
            for a in agents                        # 加回没配对的 agent
                any(i in taken for i in agent_ids(a)) || push!(new_agents, a)
            end
            agents = new_agents
        end
    end

    # (22) 解散检查, only where dissolving does not lower resilience: a coalition whose members alone all have at least its
    # margin is dissolved if its first-step cost is above that of its members alone
    final_agents = Vector{Any}()
    for a in agents
        if !outage && length(agent_ids(a)) > 1 && minimum(plans[i][2] for i in a) >= plans[a][2] - RES_TOL &&
           sum(dec_single_vals[i] for i in a) < first_step_cost(plans[a][1])
            append!(final_agents, a)               # its members act alone
        else
            push!(final_agents, a)
        end
    end
    return final_agents, [plans[a][1] for a in final_agents], num_iters
end
