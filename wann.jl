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

	function calc_rewards(
			ind::Ind,
			input::Matrix{<:T},
			ans::Matrix{<:T},
			shared_weights::Vector{<:T})::Vector{T} where T <: AbstractFloat
		n_run = length(shared_weights)
		rewards = T[]
		for w in shared_weights
			result = calc_output(ind, input, w)
			reward = -sum((result .- ans).^2)
			append!(rewards, reward)
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

	calc_rewards(ind::Ind, input::Matrix{<:AbstractFloat}, ans::Matrix{<:AbstractFloat}) =
		calc_rewards(ind, input, ans, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])

	function mutate!(ind::Ind)
		r = rand()
		if r < 0.5
			# println(i, "addconn")
			# println(ind.w)
			check_regal_matrix(ind)
			try
				mutate_addconn!(ind)
			catch
				println("no room")
				r = 1
			end
			# println_matrix(ind.w)
			check_regal_matrix(ind)
		elseif r < 0.6
			# println(i, "addnode")
			# println(ind.w)
			check_regal_matrix(ind)
			mutate_addnode!(ind)
			# println_matrix(ind.w)
			check_regal_matrix(ind)
		elseif r < 0.8
			mutate_act!(ind)
			check_regal_matrix(ind)
		end
	end

	function rank!(inds::Vector{Ind})
		# Compile objectives
		reward_avg = [ind.reward_avg for ind in inds]
		reward_max = [maximum(ind.rewards) for ind in inds]
		nConns = [length(findall(!iszero, ind.w)) for ind in inds]
		objectives = hcat(reward_avg, reward_max, 1 ./ nConns) # Maximize
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

	function train(pop::Pop, data, ans, loop, hyp)
		for i = 1:loop
			println("gen ", i)
			# result = zeros(Float64, axes(run(pop.inds[begin], data)))
			for i in 1:length(pop.inds)
				rewards = calc_rewards(pop.inds[i], data, ans)
				pop.inds[i].reward_avg = mean(rewards)
				pop.inds[i].rewards = deepcopy(rewards)
			end
			sort!(pop.inds, lt = (a, b) -> a.reward_avg < b.reward_avg)
			println("reward 1 ", pop.inds[end].reward_avg)
			# println("reward 2 ", pop.inds[end-1].reward_avg)
			# println("reward 3 ", pop.inds[end-2].reward_avg)
			rank!(pop.inds)
			n_pop = length(pop.inds)
			parents = pop.inds
			children = Vector{Ind}(undef, n_pop)

			# Sort by rank
			sort!(pop.inds, lt = (a, b) -> a.rank < b.rank)

			# Cull  - eliminate worst individuals from breeding pool
			parents = parents[1:end - floor(Int64, hyp["select_cull_ratio"] * n_pop)]

			# Elitism - keep best individuals unchanged
			n_elites = floor(Int64, hyp["select_elite_ratio"] * n_pop)
			for i in 1:n_elites
				children[i] = pop.inds[i]
			end

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
			continue
		end
	end
end

if abspath(PROGRAM_FILE) == @__FILE__
	using LinearAlgebra: transpose!
	using Flux: onehot
	using Flux.Data.MNIST
	using DataFrames # select
	using CSV # read

	function main()
		dataframe = CSV.read("data/sphere2.csv", header=true, delim=",")
		hyp = Dict(
			"select_cull_ratio" => 0.2,
			"select_elite_ratio"=> 0.2,
			"select_tourn_size" => 32,
			"prob_crossover" => 0.0
		)
		in = convert(Matrix, select(dataframe, r"i"))
		ans = convert(Matrix, select(dataframe, r"o"))
		pop = WANN.Pop(size(in, 2), size(ans, 2), 100)
		WANN.train(pop, in, ans, 15, hyp)
	end

	main()
end
