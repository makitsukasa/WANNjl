module WANN
	using Base.Threads
	using LinearAlgebra: dot
	using Statistics: mean, normalize
	using Flux
	include("./algorithm.jl")
	export Ind, WANN

	# individual
	mutable struct Ind
		nIn::Int64
		nOut::Int64
		w::Matrix{Float64} # weight
		a::Vector{<:Act} # activation function
		u::Vector{CartesianIndex{2}}
		rewards::Vector{Float64}
		reward_avg::Float64
		rank::Int64
	end

	function Base.getproperty(ind::Ind, sym::Symbol)
		if sym === :nNode
			return size(ind.w, 1)
		elseif sym === :nHid
			return ind.nNode - ind.nIn - 1 - ind.nOut
		else
			return getfield(ind, sym)
		end
	end

	Ind(nIn::Int64, nOut::Int64, w::Matrix{Float64}, a::Vector{<:Act}, u::Vector{CartesianIndex{2}}) =
		Ind(nIn,
			nOut,
			w,
			a,
			u,
			Float64[],
			NaN,
			2^63-1)

	function Ind(nIn::Int64, nOut::Int64, prob_enable::AbstractFloat)
		n = nIn + 1 + nOut
		ind = Ind(
			nIn,
			nOut,
			zeros(Float64, n, n),
			[ActOrig() for _ in 1:n],
			CartesianIndex{2}[])
		init_addconn!(ind.w, nIn + 1, prob_enable)
		return ind
	end

	Base.hash(i::Ind, h::UInt) = hash(i.a, hash(i.w, hash(:Ind, h)))
	Base.:(==)(a::Ind, b::Ind) = Base.isequal(hash(a), hash(b))
	copy(ind::Ind) = Ind(ind.nIn, ind.nOut, deepcopy(ind.w), deepcopy(ind.a), deepcopy(ind.u))

	check_regal_matrix(ind::Ind) = check_regal_matrix(ind.w, ind.nIn + 1, ind.nHid)

	function mutate_addconn!(ind::Ind)
		ind.w, ind.a, ind.u = mutate_addconn(ind.w, ind.a, ind.u, ind.nIn + 1, ind.nHid, ind.nOut)
	end

	function mutate_reviveconn!(ind::Ind)
		ind.w, ind.a, ind.u = mutate_reviveconn(ind.w, ind.a, ind.u, ind.nIn + 1, ind.nHid, ind.nOut)
	end

	function mutate_addnode!(ind::Ind)
		ind.w, ind.a, ind.u = mutate_addnode(ind.w, ind.a, ind.u, ind.nIn + 1)
	end

	function mutate_act!(ind::Ind)
		connected_indices = findall(!iszero, reshape(sum(ind.w, dims = 2), size(ind.w, 2)))
		ind.a = mutate_act(ind.a, connected_indices)
	end

	function make_onehot(a::Vector{<:Number})
		index = findall(a .== maximum(a))
		# println("onehot index :", index)
		return [in(i, index) ? 1.0 : 0.0 for i in 1:length(a)]
	end

	function calc_output(ind::Ind, input::Matrix{<:AbstractFloat}, shared_weight::AbstractFloat)
		buff = zeros((axes(input, 1), ind.nNode))
		buff[:, 1] .= 1 # bias
		buff[:, 2:ind.nIn+1] = input
		for i in (ind.nIn + 2):ind.nNode
			b = buff * ind.w[:, i]
			buff[:, i] = call(ind.a[i], b) .* shared_weight
		end
		return buff[:, (end - ind.nOut + 1):end]
	end

	function calc_output_(ind::Ind, input::Matrix{<:AbstractFloat}, shared_weight::AbstractFloat)
		buff = zeros((axes(input, 1), ind.nNode))
		buff[:, 1] .= 1 # bias
		buff[:, 2:ind.nIn + 1] = input
		l = ind.nIn + 2
		while l <= ind.nNode
			r = l
			while r <= ind.nNode
				if iszero(ind.w[l:r, l:r])
					r += 1
					continue
				end
				r -= 1
				break
			end
			r = min(r, ind.nNode)
			if l == r
				b = buff * ind.w[:, l]
				buff[:, l] = b .* shared_weight
			else
				b = buff * ind.w[:, l:r]
				for i in l:r
					buff[:, i] .= b[i - l + 1] .* shared_weight
				end
			end
			if r == ind.nNode
				break
			end
			l = r + 1
		end
		return buff[:, (end - ind.nOut + 1):end]
	end

	function calc_rewards(
			ind::Ind,
			reward::Function,
			input::Matrix{<:T},
			ans::Matrix{<:T},
			shared_weights::Vector{<:T})::Vector{T} where T <: AbstractFloat
		n_run = length(shared_weights)
		n_sample = size(input, 1)
		rewards = Vector{T}(undef, n_run)
		for i in 1:n_run
			result = calc_output(ind, input, shared_weights[i])
			# result_ = calc_output_(ind, input, shared_weights[i])
			# if !isapprox(result, result_)
			# 	println("calc_output is wrong")
			# 	open("input.txt", "w") do fp
			# 		println(fp, input)
			# 	end
			# 	open("ind.w.txt", "w") do fp
			# 		println(fp, ind.w)
			# 	end
			# 	exit()
			# end
			rewards[i] = reward(result, ans)
			# if rewards[i] == -1.0
			# 	# println("input        : ", input)
			# 	println("result       : ", result)
			# 	println("ans          : ", ans)
			# 	println("result .- ans: ", result .- ans)
			# 	println("abs          : ", abs.(result .- ans))
			# 	println("norm(abs)    : ", abs.(result .- ans) ./ ans)
			# 	println("mean(abs)    : ", mean(abs.(result .- ans) ./ ans))
			# 	println("reward       : ", rewards[i])
			# 	exit()
			# end
		end
		return rewards
	end

	calc_rewards(ind::Ind, reward::Function, input::Matrix{<:AbstractFloat}, ans::Matrix{<:AbstractFloat}) =
		calc_rewards(ind, reward, input, ans, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])

	function mutate!(ind::Ind, prob_addconn, prob_reviveconn, prob_addnode, prob_mutateact)
		r = rand()
		if r < prob_addconn
			try
				mutate_addconn!(ind)
				# check_regal_matrix(ind)
			catch e
				if hasfield(typeof(e), :msg) && e.msg == "no room"
					# println("no room for connect")
					# mutate!(ind, prob_addconn, prob_reviveconn, prob_addnode, prob_mutateact)
				else
					rethrow(e)
				end
			end
		elseif r < prob_addconn + prob_reviveconn
			try
				# println_matrix(ind.w)
				# before = deepcopy(ind.w)
				mutate_reviveconn!(ind)
				# check_regal_matrix(ind)
				# println_matrix(before)
				# println()
				# println_matrix(ind.w)
				# println()
				# exit()
			catch e
				if hasfield(typeof(e), :msg) && e.msg == "could not revive"
					# println("no candidate for revive")
					# mutate!(ind, prob_addconn, prob_reviveconn, prob_addnode, prob_mutateact)
				else
					rethrow(e)
				end
			end
		elseif r < prob_addconn + prob_reviveconn + prob_addnode
			try
				# before = deepcopy(ind.u)
				mutate_addnode!(ind)
				# println("before: ", before)
				# println("after : ", ind.u)
				# check_regal_matrix(ind)
			catch e
				if hasfield(typeof(e), :msg) && e.msg == "no candidate found"
					# println("no connect for insert")
					# mutate!(ind, prob_addconn, prob_reviveconn, prob_addnode, prob_mutateact)
				else
					rethrow(e)
				end
			end
		elseif r < prob_addconn + prob_reviveconn + prob_addnode + prob_mutateact
			try
				mutate_act!(ind)
				# check_regal_matrix(ind)
			catch e
				if hasfield(typeof(e), :msg) && e.msg == "no connect"
					# println("no connect for mutate act")
					# mutate!(ind, prob_addconn, prob_reviveconn, prob_addnode, prob_mutateact)
				else
					rethrow(e)
				end
			end
		end
	end

	function rank!(inds::Vector{Ind}, alg_probMoo)
		shuffle!(inds)
		# Alternate second objective
		if  rand() > alg_probMoo
			# Compile objectives
			reward_avg = [ind.reward_avg for ind in inds]
			reward_max = [maximum(ind.rewards) for ind in inds]
			objectives = hcat(normalize(reward_avg), normalize(reward_max))
		else
			# Compile objectives
			reward_avg = [ind.reward_avg for ind in inds]
			nConns = [length(findall(!iszero, ind.w)) for ind in inds]
			objectives = hcat(normalize(reward_avg), normalize(1 ./ nConns)) # Maximize
		end

		rank = non_dominated_sort(objectives)
		for i in 1:length(rank)
			inds[i].rank = rank[i]
		end
	end


	mutable struct Pop
		inds::Array{Ind}
	end

	Base.getindex(pop::Pop, index) = getindex(pop.inds, index)

	Base.copy(pop::Pop) = Pop(copy(pop.inds))

	function Pop(nIn::Integer, nOut::Integer, size::Integer, prob_enable::AbstractFloat)
		return Pop([Ind(nIn, nOut, prob_enable) for i = 1:size])
	end

	function test(pop::Pop, test_func, data, ans)
		# sort!(pop.inds, lt = (a, b) -> a.reward_avg > b.reward_avg)
		# ind = pop.inds[1]
		outputs = []
		for ind in pop.inds
			for w in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
				o = calc_output(ind, data, w)
				push!(outputs, o)
			end
		end
		test_func(outputs, ans)
	end

	function test(pop::Pop, test_func, data, ans, w)
		# sort!(pop.inds, lt = (a, b) -> a.reward_avg > b.reward_avg)
		# ind = pop.inds[1]
		outputs = []
		for ind in pop.inds
			o = calc_output(ind, data, w)
			push!(outputs, o)
		end
		test_func(outputs, ans)
	end

	function train(param)
		pop = deepcopy(param["pop"])
		data = param["data"]
		ans = param["ans"]
		test_data = param["test_data"]
		test_ans = param["test_ans"]
		hyp = param["hyp"]
		reward = param["reward"]
		test_func = param["test"]
		for i = 1:param["n_generation"]
			println("gen $i")
			# result = zeros(Float64, axes(run(pop.inds[begin], data)))
			for i in 1:length(pop.inds)
				rewards = calc_rewards(pop.inds[i], reward, data, ans)
				pop.inds[i].reward_avg = mean(rewards)
				pop.inds[i].rewards = deepcopy(rewards)
				if isnan(sum(rewards)) || isinf(sum(rewards))
					println("reward is invalid $rewards")
					open("hoge.txt", "w") do fp
						write(fp, "reward is invalid $rewards\n")
						write(fp, "w\n")
						println_matrix(fp, pop.inds[i].w)
						write(fp, "a\n")
						println_matrix(fp, pop.inds[i].a)
						write(fp, "data\n")
						println_matrix(fp, data)
						for w in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
							write(fp, "output when weight is $w\n")
							println_matrix(fp, calc_output(pop.inds[i], data, w))
						end
						write(fp, "ans\n")
						println_matrix(fp, ans)
					end
				end
			end
			sort!(pop.inds, lt = (a, b) -> a.reward_avg > b.reward_avg)
			println("reward $(pop.inds[1].reward_avg), $(pop.inds[2].reward_avg), $(pop.inds[3].reward_avg)")
			# println_matrix(pop.inds[1].w)
			# println([a.id for a in pop.inds[1].a])
			# println()
			# println_matrix(pop.inds[2].w)
			# println([a.id for a in pop.inds[2].a])
			# println()
			# println_matrix(pop.inds[3].w)
			# println([a.id for a in pop.inds[3].a])
			# println()

			# if i in vcat([1, 10, 20, 30, 40, 50, 70, collect(100:100:10000)]...)
			# if true
			if false
				print("test for train data, ")
				test(pop, test_func, data, ans)
				print("test for test  data, ")
				test(pop, test_func, test_data, test_ans)
			end

			rank!(pop.inds, hyp["alg_probMoo"])
			n_pop = length(pop.inds)
			parents = deepcopy(pop.inds)
			children = Vector{Ind}(undef, n_pop)

			# Sort by rank
			sort!(parents, lt = (a, b) -> a.rank < b.rank)

			# Elitism - keep best individuals unchanged
			n_elites = floor(Int64, hyp["select_elite_ratio"] * n_pop)
			children[1:n_elites] = deepcopy(parents[1:n_elites])

			# Cull  - eliminate worst individuals from breeding pool
			n_cull = floor(Int64, hyp["select_cull_ratio"] * n_pop)
			parents = parents[1:end-n_cull]

			# Get parent pairs via tournament selection
			# -- As individuals are sorted by fitness, index comparison is
			# enough. In the case of ties the first individual wins
			n_generate = n_pop - n_elites
			parentA = rand(1:length(parents), n_generate, hyp["select_tourn_size"])
			parentB = rand(1:length(parents), n_generate, hyp["select_tourn_size"])
			parentAB = transpose(hcat(minimum(parentA, dims=2), minimum(parentB, dims=2)))
			# println(size(parentA))
			# println_matrix(parentA)
			# println(size(parentB))
			# println_matrix(parentB)
			# println(size(parentAB))
			# println_matrix(parentAB)
			# exit()
			sort!(parentAB, dims=1)

			# Breed child population
			@threads for i in 1:n_generate
				if hyp["prob_crossover"] <= rand()
					# Mutation only: take only highest fit parent
					child = copy(parents[parentAB[1, i]])
				else
					# Crossover
					throw(error("crossover is not impremented"))
				end

				mutate!(child, hyp["prob_addconn"], hyp["prob_reviveconn"], hyp["prob_addnode"], hyp["prob_mutateact"])
				children[n_elites + i] = child
			end

			pop.inds = deepcopy(children)

			# cnt = 0
			# for a in 2:length(pop.inds)
			# 	for b in 1:a-1
			# 		if pop.inds[a] == pop.inds[b]
			# 			cnt += 1
			# 			println_matrix(pop.inds[a].w)
			# 			println_matrix(pop.inds[b].w)
			# 			println()
			# 		end
			# 	end
			# end
			# if cnt > 0
			# 	println("$cnt same inds")
			# 	if cnt >= length(pop.inds)
			# 		exit()
			# 	end
			# end
		end
		println()
		println("test for train data")
		for w in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
			print("w is $w, ")
			test(pop, test_func, data, ans, w)
		end
		println("test for test  data")
		for w in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
			print("w is $w, ")
			test(pop, test_func, test_data, test_ans, w)
		end

		for i in 1:length(pop.inds)
			rewards = calc_rewards(pop.inds[i], reward, data, ans)
			pop.inds[i].reward_avg = mean(rewards)
			pop.inds[i].rewards = deepcopy(rewards)
		end
		sort!(pop.inds, lt = (a, b) -> a.reward_avg > b.reward_avg)
		return pop.inds[1]
	end
end

if abspath(PROGRAM_FILE) == @__FILE__
	using DataFrames # select
	using CSV # read
	using Statistics: mean
	include("./algorithm.jl")

	function main()
		n_pop = 100
		n_sample = 100
		n_generation = 1000

		function reward(output, ans)
			return -mean(abs.(ans .- output) ./ ans)
		end

		function test(outputs, ans)
			n_test = size(ans, 1)
			diffs = []
			for o in outputs
				push!(diffs, mean(abs.(ans .- o) ./ ans))
			end
			println("avg diff : $(mean(diffs)), min diff : $(minimum(diffs))")
		end

		dataframe = CSV.read("domain/sphere/data/sphere5.csv", header=true, delim=",")
		n_sample = min(n_sample, div(size(dataframe)[1], 5) * 4)
		n_test = div(n_sample, 5)
		in = convert(Matrix, select(dataframe, r"i"))
		ans = convert(Matrix, select(dataframe, r"o"))
		test_in = in[(n_sample - n_test + 1):n_sample, :]
		test_ans = ans[(n_sample - n_test + 1):n_sample, :]
		in = in[1:(n_sample - n_test), :]
		ans = ans[1:(n_sample - n_test), :]

		hyp = Dict(
			"select_cull_ratio" => 0.2,
			"select_elite_ratio"=> 0.2,
			"select_tourn_size" => 32,
			"prob_initEnable" => 0.05,
			"alg_probMoo" => 0.8,
			"prob_crossover" => 0.0,
			"prob_addnode" => 0.2,
			"prob_reviveconn" => 0.05,
			"prob_addconn" => 0.45,
			"prob_mutateact" => 0.3,
		)

		param_for_train = Dict(
			"pop" => WANN.Pop(size(in, 2), size(ans, 2), n_pop, hyp["prob_initEnable"]),
			"data" => in,
			"ans" => ans,
			"test_data" => test_in,
			"test_ans" => test_ans,
			"n_generation" => n_generation,
			"reward" => reward,
			"test" => test,
			"hyp" => hyp,
		)

		println("train")
		WANN.train(param_for_train)
	end

	main()
end
