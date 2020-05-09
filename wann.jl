module WANN
	using LinearAlgebra: dot
	using Flux
	include("./algorithm.jl")
	export Ind, WANN

	mutable struct Ind
		nIn::Integer
		nOut::Integer
		loss::Float64
		v::Array{Float64,2}
	end

	function Base.getproperty(ind::Ind, sym::Symbol)
		if sym === :nNode
			return size(ind.v, 1)
		elseif sym === :nHid
			return ind.nNode - ind.nIn - 1 - ind.nOut
		else
			return getfield(ind, sym)
		end
	end

	copy(ind::Ind) = Ind(ind.nIn, ind.nOut, ind.loss, deepcopy(ind.v))

	check_regal_matrix(ind::Ind) = check_regal_matrix(ind.v, ind.nIn + 1, ind.nHid)

	function Ind(nIn::Integer, nOut::Integer)
		n = nIn + 1 + nOut
		ind = Ind(
			nIn,
			nOut,
			2.0^64,
			zeros(Float64, (n, n)))
		ind.v = mutate_addconn(ind)
		return ind
	end

	mutate_addconn(ind::Ind)::Array{Float64, 2} =
		mutate_addconn(ind.v, ind.nIn + 1, ind.nHid, ind.nOut)

	mutate_addnode(ind::Ind)::Array{Float64, 2} =
		mutate_addnode(ind.v, ind.nIn + 1)

	function run(ind::Ind, input::Array{Float64, 2})
		buff = zeros((axes(input, 1), ind.nNode))
		# println("axes(buff): ", axes(buff))
		# println("axes(buff[1, :]): ", axes(buff[:, 1]))
		# println("axes(buff[2:ind.nIn+1, :]): ", axes(buff[:, 2:ind.nIn+1]))
		buff[:, 1] .= 1 # bias
		buff[:, 2:ind.nIn+1] = input
		for i = ind.nIn+2:ind.nNode
			b = buff * ind.v[:, i]
			# b = activation_function(b)
			buff[:, i] = b
		end
		return buff[:, end-ind.nOut+1:end]
	end


	mutable struct Pop
		inds::Array{Ind}
	end

	copy(pop::Pop) = Pop(copy(pop.inds))

	function Pop(nIn::Integer, nOut::Integer, size::Integer)
		return Pop([Ind(nIn, nOut) for i = 1:size])
	end

	function train(pop::Pop, data, ans, loop)
		for i = 1:loop
			println("gen ", i)
			# result = zeros(Float64, axes(run(pop.inds[begin], data)))
			for i in 1:length(pop.inds)
				result = run(pop.inds[i], data)
				# println("result .- ans: ", result .- ans)
				# println("sum(lost): ",sum(abs.(result .- ans)))
				pop.inds[i].loss = sum(abs.(result .- ans))
			end
			sort!(pop.inds, lt = (a, b) -> a.loss < b.loss)
			println("loss: ", pop.inds[1].loss, ", ",
				pop.inds[2].loss, ", ",
				pop.inds[3].loss, ", ",
				pop.inds[4].loss)
			newInds = [copy.(pop.inds[1:div(length(pop.inds), 2)])
				copy.(pop.inds[1:div(length(pop.inds), 2)])] |> vcat
			# println("")
			# for i in 1:length(pop.inds)
			# 	println(pop.inds[i].v)
			# end
			println("lost ", pop.inds[1].loss)
			for i in 1:length(newInds)
				r = rand()
				if r < 0.5
					# println(i, "addconn")
					# println(newInds[i].v)
					check_regal_matrix(newInds[i])
					newInds[i].v = mutate_addconn(newInds[i])
					# println_matrix(newInds[i].v)
					check_regal_matrix(newInds[i])
				elseif r < 0.6
					# println(i, "addnode")
					# println(newInds[i].v)
					check_regal_matrix(newInds[i])
					newInds[i].v = mutate_addnode(newInds[i])
					# println_matrix(newInds[i].v)
					check_regal_matrix(newInds[i])
				end
			end
			# println(typeof(pop.inds), axes(pop.inds))
			# println(typeof(newInds), axes(newInds))
			# println("")
			# for i in 1:length(newInds)
			# 	println(newInds[i].v)
			# end
			pop.inds = newInds
			println("")
		end
	end
end

if abspath(PROGRAM_FILE) == @__FILE__
	using LinearAlgebra: transpose!
	using Flux: onehot
	using Flux.Data.MNIST

	function main()
		in = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0; 0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0;]
		# ans = [0.0 0.0 0.0 1.0; 0.0 1.0 1.0 1.0; 0.0 1.0 1.0 0.0; 1.0 0.0 0.0 1.0;
		#        0.0 0.0 0.0 1.0; 0.0 1.0 1.0 1.0; 0.0 1.0 1.0 0.0; 1.0 0.0 0.0 1.0;]
		ans = [0.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 1.0; 0.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 1.0]
		pop = WANN.Pop(2, 2, 4)
		WANN.train(pop, in, ans, 100)
	end

	main()
end
