module WANN
	using LinearAlgebra: dot
	using Statistics: mean
	using Flux
	include("./algorithm.jl")
	export Ind, WANN

	# individual
	mutable struct Ind
		nIn::Int64
		nOut::Int64
		w::Matrix{Float64} # weight
		a::Vector{<:Act} # activation function
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

	Ind(nIn::Int64, nOut::Int64, w::Matrix{Float64}, a::Vector{<:Act}) =
		Ind(nIn,
			nOut,
			deepcopy(w),
			deepcopy(a),
			Float64[],
			NaN,
			2^63-1)

	function Ind(nIn::Int64, nOut::Int64, prob_enable::AbstractFloat)
		n = nIn + 1 + nOut
		ind = Ind(
			nIn,
			nOut,
			zeros(Float64, n, n),
			[ActOrig() for _ in 1:n])
		init_addconn!(ind.w, nIn + 1, prob_enable)
		return ind
	end

	copy(ind::Ind) = Ind(ind.nIn, ind.nOut, deepcopy(ind.w), deepcopy(ind.a))

	check_regal_matrix(ind::Ind) = check_regal_matrix(ind.w, ind.nIn + 1, ind.nHid)

	function mutate_addconn!(ind::Ind)
		ind.w, ind.a = mutate_addconn(ind.w, ind.a, ind.nIn + 1, ind.nHid, ind.nOut)
	end

	function mutate_addnode!(ind::Ind)
		ind.w, ind.a = mutate_addnode(ind.w, ind.a, ind.nIn + 1)
	end

	function mutate_act!(ind::Ind)
		ind.a = mutate_act(ind.a)
	end

	function make_onehot(a::Vector{<:Number})
		index = findall(a .== maximum(a))
		# println("onehot index :", index)
		return [in(i, index) ? 1.0 : 0.0 for i in 1:length(a)]
	end

	function calc_output(ind::Ind, input::Matrix{<:AbstractFloat}, shared_weight::AbstractFloat)
		buff = zeros((axes(input, 1), ind.nNode))
		# println("axes(buff): ", axes(buff))
		# println("axes(buff[1, :]): ", axes(buff[:, 1]))
		# println("axes(buff[2:ind.nIn+1, :]): ", axes(buff[:, 2:ind.nIn+1]))
		buff[:, 1] .= 1 # bias
		buff[:, 2:ind.nIn+1] = input
		for i in ind.nIn+2:ind.nNode
			b = buff * ind.w[:, i]
			# println(size(buff), size(ind.w[:, i]), size(b))
			buff[:, i] = call(ind.a[i], b) * shared_weight
		end
		return buff[:, end-ind.nOut+1:end]
	end

	# function classify(ind::Ind,
	# 			input::Matrix{<:AbstractFloat},
	# 			shared_weights::Vector{<:AbstractFloat} = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
	# 	ans = zeros(size(calc_output(ind, input, shared_weights[1])))
	# 	for w in shared_weights
	# 		o = calc_output(ind, input, w)
	# 		softmax = mapslices(x -> exp.(x) ./ sum(exp.(x)), o, dims = 1)
	# 		# println("before softmax :", o)
	# 		# println("softmax dim1 :", mapslices(make_onehot, o, dims = 1))
	# 		# println("softmax dim2 :", mapslices(make_onehot, o, dims = 2))
	# 		ans += softmax
	# 	end
	# 	# println("before classify :", ans)
	# 	# println("after  classify :", ret)
	# 	# ans = mapslices(make_onehot, ans, dims = 2)
	# 	# ret = mapslices(x -> x ./ length(findall(!iszero, x)), ans, dims = 2)
	# 	return ans
	# end

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
			rewards[i] = reward(result, ans)
			# println("input        : ", input)
			# println("result       : ", result)
			# println("ans          : ", ans)
			# println("result .- ans: ", result .- ans)
			# println("square       : ", (result .- ans).^2)
			# println("sum          : ", sum((result .- ans).^2))
			# println("reward       : ", reward)
		end
		return rewards
	end

	calc_rewards(ind::Ind, reward::Function, input::Matrix{<:AbstractFloat}, ans::Matrix{<:AbstractFloat}) =
		calc_rewards(ind, reward, input, ans, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])

	function mutate!(ind::Ind)
		r = rand()
		if r < 0.25
			# println(i, "addconn")
			# println(ind.w)
			check_regal_matrix(ind)
			try
				mutate_addconn!(ind)
			catch
				println("no room")
				# println_matrix(ind.w)
				r = 1.0
			end
			# println_matrix(ind.w)
			check_regal_matrix(ind)
		elseif r < 0.5
			# println(i, "addnode")
			# println(ind.w)
			check_regal_matrix(ind)
			mutate_addnode!(ind)
			# println_matrix(ind.w)
			check_regal_matrix(ind)
		elseif r < 1.0
			mutate_act!(ind)
			check_regal_matrix(ind)
		end
	end

	function rank!(inds::Vector{Ind}, alg_probMoo)
		# Compile objectives
		reward_avg = [ind.reward_avg for ind in inds]
		reward_max = [maximum(ind.rewards) for ind in inds]
		nConns = [length(findall(!iszero, ind.w)) for ind in inds]

		# Alternate second objective
		if  rand() > alg_probMoo
			objectives = hcat(reward_avg, reward_max)
		else
			objectives = hcat(reward_avg, 1 ./ nConns) # Maximize
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

	function evolve_pop()

	end

	function mutate!(inds::Vector{<:Ind})
		for i in 1:length(inds)
			mutate!(i)
		end
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

	function train(param)
		pop = param["pop"]
		data = param["data"]
		ans = param["ans"]
		test_data = param["test_data"]
		test_ans = param["test_ans"]
		hyp = param["hyp"]
		reward = param["reward"]
		test_func = param["test"]
		for i = 1:param["n_generation"]
			println("gen ", i)
			# result = zeros(Float64, axes(run(pop.inds[begin], data)))
			for i in 1:length(pop.inds)
				rewards = calc_rewards(pop.inds[i], reward, data, ans)
				pop.inds[i].reward_avg = mean(rewards)
				pop.inds[i].rewards = deepcopy(rewards)
			end
			sort!(pop.inds, lt = (a, b) -> a.reward_avg > b.reward_avg)
			println("reward 1 ", pop.inds[1].reward_avg)
			# println("reward 2 ", pop.inds[2].reward_avg)
			# println("reward 3 ", pop.inds[3].reward_avg)

			# if i in vcat([collect(1:50), collect(100:100:10000)]...)
			if true
				print("test for train data, ")
				test(pop, test_func, data, ans)
				print("test for test  data, ")
				test(pop, test_func, test_data, test_ans)
			end

			rank!(pop.inds, hyp["alg_probMoo"])
			n_pop = length(pop.inds)
			parents = pop.inds
			children = Vector{Ind}(undef, n_pop)

			# Sort by rank
			sort!(pop.inds, lt = (a, b) -> a.rank < b.rank)

			# Elitism - keep best individuals unchanged
			n_elites = floor(Int64, hyp["select_elite_ratio"] * n_pop)
			for i in 1:n_elites
				children[i] = pop.inds[i]
			end

			# Cull  - eliminate worst individuals from breeding pool
			parents = parents[1:end - floor(Int64, hyp["select_cull_ratio"] * n_pop)]

			# Get parent pairs via tournament selection
			# -- As individuals are sorted by fitness, index comparison is
			# enough. In the case of ties the first individual wins
			n_generate = n_pop - n_elites
			parentA = rand(1:n_pop, n_generate, hyp["select_tourn_size"])
			parentB = rand(1:n_pop, n_generate, hyp["select_tourn_size"])
			parents = transpose(hcat(minimum(parentA, dims=2), minimum(parentB, dims=2)))
			sort!(parents, dims=1)

			# Breed child population
			for i in 1:n_generate
				if rand() > hyp["prob_crossover"]
					# Mutation only: take only highest fit parent
					child = copy(pop[parents[1, i]])
				else
					# Crossover
					throw(error("crossover is not impremented"))
				end
				mutate!(child)
				children[n_elites + i] = child
			end

			pop.inds = children
		end
		print("test for train data, ")
		test(pop, test_func, data, ans)
		print("test for test  data, ")
		test(pop, test_func, test_data, test_ans)
	end
end

if abspath(PROGRAM_FILE) == @__FILE__
	using LinearAlgebra: transpose!
	using Flux: onehot
	using Flux.Data.MNIST
	using DataFrames # select
	using CSV # read

	function main()
		dataframe = CSV.read("data/sphere5.csv", header=true, delim=",")
		hyp = Dict(
			"select_cull_ratio" => 0.2,
			"select_elite_ratio"=> 0.2,
			"select_tourn_size" => 32,
			"prob_initEnable" => 0.05,
			"alg_probMoo" => 0.8,
			"prob_crossover" => 0.0
		)
		in = convert(Matrix, select(dataframe, r"i"))
		ans = convert(Matrix, select(dataframe, r"o"))
		n_test = div(size(dataframe)[2], 5)
		in_test = in[end-n_test:end]
		ans_test = ans[end-n_test:end]
		in = in[1:end-n_test]
		ans = ans[1:end-n_test]
		pop = WANN.Pop(size(in, 2), size(ans, 2), 500, hyp["prob_initEnable"])
		println("train")
		WANN.train(pop, in, ans, in_test, ans_test, 1000, hyp)
	end

	main()
end
