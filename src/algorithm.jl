using Random: shuffle!
using Base: print_matrix
include("./act.jl")

Base.print_matrix(mat::Union{Core.AbstractArray,Core.AbstractArray}) =
	print_matrix(stdout, mat)

function println_matrix(mat::Union{Core.AbstractArray,Core.AbstractArray})
	print_matrix(stdout, mat)
	println()
end

function check_regal_matrix(v, nIn, nHid)
	if length(findall(x -> x > 0, v)) == 0
		println("irregal : no connection")
		println_matrix(v)
		throw(error("irregal : no connection"))
	end
	indices = findall(x -> x == 1, v)
	candidate = CartesianIndex{2}[]
	for c in indices
		y = c[1]
		x = c[2]
		# reverse order
		if x <= y
			println("irregal : x($x) <= y($y)")
			println_matrix(v)
			throw(error("irregal : x <= y"))
		# dst is input node
		elseif x <= nIn
			println("irregal : x($x) < nIn($nIn)")
			println_matrix(v)
			throw(error("irregal : x < nIn"))
		# src is output node
		elseif y > nIn + nHid
			println("irregal : y($y) >= nIn + nHid($nIn + $nHid)")
			println_matrix(v)
			throw(error("irregal : y >= nIn + nHid"))
		end
	end
end

# Warshall's original algorithm by a bool adjacency matrix
function warshall(adjacency_matrix)
	ans = deepcopy(adjacency_matrix)
	n = size(adjacency_matrix, 1)
	for k = 1:n, i = 1:n, j = 1:n
		ans[i, j] |= ans[i, k] && ans[k, j]
	end
	return ans
end

# Warshall's original algorithm by a bool adjacency matrix
function topological_sort(order::Vector{Int}, adjacency_matrix::Array{Bool, 2}, reachability_matrix::Array{Bool, 2})
	# http://blog.gapotchenko.com/stable-topological-sort
	n = length(order)
	ans = deepcopy(order)
    @label restart
	for i = 1:n, j = 1:i
		i_ = ans[i]
		j_ = ans[j]
		if !adjacency_matrix[i_, j_]
			continue
		end
		j_on_i = reachability_matrix[j_, i_]
		i_on_j = reachability_matrix[i_, j_]
		if j_on_i && i_on_j
			throw(error("circular way found"))
			exit(-1)
		end
		deleteat!(ans, i)
		insert!(ans, j, i_)
		@goto restart;
	end
	return ans
end

function get_shuffued_order(
		v::Matrix{<:AbstractFloat},
		nIn::Int, nOut::Int,
		orig_order::Vector{Int})::Vector{Int}
	hid = orig_order[nIn+1:end-nOut]
	shuffle!(hid)
	return [orig_order[1:nIn]; hid; orig_order[end-nOut+1:end]]
end

get_shuffued_order(v::Matrix{<:AbstractFloat}, nIn::Int, nOut::Int)::Vector{Int} =
	get_shuffued_order(v, nIn, nOut, collect(1:size(v,1)))

function get_sorted_order(
		v::Matrix{<:AbstractFloat},
		nIn::Int, nOut::Int,
		orig_order::Vector{Int})::Vector{Int}
	n = size(v, 1)
	adjacency_matrix = map(x-> x > 0 ? true : false, v)
	reachability_matrix = warshall(adjacency_matrix)
	return topological_sort(orig_order, adjacency_matrix, reachability_matrix)
	# return 1:size(v,1)
end

get_sorted_order(v::Matrix{<:AbstractFloat}, nIn::Int, nOut::Int)::Vector{Int} =
	get_sorted_order(v, nIn, nOut, collect(1:size(v,1)))

function get_assigned_v(
		orig::Matrix{<:AbstractFloat},
		order::Vector{Int},
		special::Dict{CartesianIndex{2}, <:AbstractFloat} = Dict{CartesianIndex{2}, <:AbstractFloat}(),
		default::AbstractFloat = 0.0)::Matrix{<:AbstractFloat}
	n = length(order)
	ans = zeros((n, n))
	for y = 1:n, x = 1:n
		if haskey(special, CartesianIndex(y, x))
			ans[y, x] = special[CartesianIndex(y, x)]
			continue
		end
		try
			ans[y, x] = orig[order[y], order[x]]
		catch
			ans[y, x] = default
		end
	end
	return ans
end

get_assigned_v(orig::Matrix{<:AbstractFloat},
		special::Dict{CartesianIndex{2}, <:AbstractFloat} = Dict{CartesianIndex{2}, <:AbstractFloat}(),
		default::AbstractFloat = 0.0)::Matrix{<:AbstractFloat} =
	get_assigned_v(orig, collect(1:size(orig, 1)), special, default)

function get_assigned_a(
		orig::Vector{T},
		order::Vector{Int},
		default::Function = () -> ActOrig())::Vector{T} where T<:Act
	n = length(order)
	ans = Array{T}(undef, n)
	for i = 1:n
		try
			ans[i] = orig[order[i]]
		catch
			ans[i] = default()
		end
	end
	return ans
end

function get_random_index(f::Function, v::Matrix{<:AbstractFloat})::CartesianIndex{2}
	candidate = findall(f, v)
	if length(candidate) == 0
		println("no candidate found")
		println_matrix(v)
		throw(error("no candidate found"))
	end
	return candidate[rand(1:length(candidate))]
end

function get_all_connectable_indices(
		v::Matrix{<:AbstractFloat},
		nIn::Int,
		nHid::Int,
		order::Vector{Int})::Vector{CartesianIndex{2}}
	indices = findall(x -> x == 0, v)
	ans = CartesianIndex{2}[]
	for c in indices
		y = order[c[1]]
		x = order[c[2]]
		# print("($x, $y) : ")
		# ignore non-connectable
		# reverse order
		if x <= y
			# println("x <= y")
			continue
		# dst is input node
		elseif x <= nIn
			# println("x < nIn($nIn)")
			continue
		# src is output node
		elseif y > nIn + nHid
			# println("y >= nIn + nHid($nIn + $nHid)")
			continue
		end
		# println("pushed")
		push!(ans, CartesianIndex(y, x))
	end
	if length(ans) == 0
		throw(error("no room"))
	end
	return ans
end

function get_random_connectable_index(
		v::Matrix{<:AbstractFloat},
		nIn::Int,
		nHid::Int,
		order::Vector{Int})::CartesianIndex{2}
	candidate = get_all_connectable_indices(v, nIn, nHid, order)
	return candidate[rand(1:length(candidate))]
end

get_random_connectable_index(
		v::Matrix{<:AbstractFloat},
		nIn::Int,
		nHid::Int)::CartesianIndex{2} =
	get_random_connectable_index(v, nIn, nHid, collect(1:size(v,1)))

function init_addconn!(v::Matrix{<:AbstractFloat}, nIn, prob_enable)
	candidate = get_all_connectable_indices(v, nIn, 0, collect(1:size(v, 1)))
	for index in candidate
		if rand() < prob_enable
			v[index] = 1
		end
	end
	if length(findall(x -> x > 0, v)) == 0
		index = get_random_connectable_index(v, nIn, 0, collect(1:size(v, 1)))
		v[index] = 1
	end
end

function mutate_addconn(
		v::Matrix{<:AbstractFloat},
		a::Vector{<:Act},
		nIn::Int,
		nHid::Int,
		nOut::Int)::Tuple{Matrix{<:AbstractFloat}, Vector{<:Act}}
	# shuhhle
	order = get_shuffued_order(v, nIn, nOut)
	# println("shuffled: ", order)
	# sort
	order = get_sorted_order(v, nIn, nOut, order)
	# println("sorted:   ", order)
	# index
	index = get_random_connectable_index(v, nIn, nHid, order)
	# println("connect:   ", index)
	# assign
	return get_assigned_v(v, order, Dict([(index, 1.0)])), get_assigned_a(a, order)
end

function mutate_addnode(
		v::Matrix{<:AbstractFloat},
		a::Vector{<:Act},
		nIn::Int)::Tuple{Matrix{<:AbstractFloat}, Vector{<:Act}}
	# index
	index = get_random_index(x -> x == 1, v)
	src = index[1]
	dst = index[2]
	new_node_index = nIn > src ? nIn + 1 : src + 1

	# order = [1, 2, 3, 0, 4, 5]
	order = collect(1:size(v, 1))
	insert!(order, new_node_index, 0)

	# disable src->dst, enable src->new, new->dst
	special = Dict{CartesianIndex{2}, AbstractFloat}(
		CartesianIndex(src, dst+1) => 0.0,
		CartesianIndex(src, new_node_index) => 1.0,
		CartesianIndex(new_node_index, dst+1) => 1.0,
	)
	# println("src:$src, dst:$dst, new:$new_node_index")

	# assign
	return get_assigned_v(v, order, special), get_assigned_a(a, order)
end

function mutate_act(a::Vector{<:Act}, mutable_indices)
	mutate!(a[mutable_indices[rand(1:end)]])
	return a
end

function get_fronts(objectives::Matrix{<:AbstractFloat})::Vector{Vector{Any}}
	values1 = objectives[1:end,1]
	values2 = objectives[1:end,2]
	S=[[] for i in 1:length(values1)]
	front = [[]]
	n=[0 for i in 1:length(values1)]
	rank = [0 for i in 1:length(values1)]

	# Get dominance relations
	for p = 1:length(values1)
		for q = 1:length(values1)
			if (values1[p] >  values1[q] && values2[p] >  values2[q]) ||
			   (values1[p] >= values1[q] && values2[p] >  values2[q]) ||
			   (values1[p] >  values1[q] && values2[p] >= values2[q])
			  if !(q in S[p])
				push!(S[p], q)
			  end
			elseif (values1[q] >  values1[p] && values2[q] >  values2[p]) ||
				   (values1[q] >= values1[p] && values2[q] >  values2[p]) ||
				   (values1[q] >  values1[p] && values2[q] >= values2[p])
				n[p] = n[p] + 1
			end
		end
		if n[p]==0
			rank[p] = 0
			if !(p in front[1])
				push!(front[1], p)
			end
		end
	end

	# Assign front
	i = 1
	while front[i] != []
		Q=[]
		for p in front[i]
			for q in S[p]
				n[q] = n[q] - 1
				if n[q] != 0
				  continue
				end
				rank[q] = i + 1
				if !(q in Q)
				  # println(q)
				  push!(Q, q)
				end
			end
		end
		i = i+1
		push!(front, Q)
		# println(Q)
		# println(front)
	end
	return front[1:end-1] # front[end] is []
end

function get_crowding_dist(objectives::Vector{T})::Vector{T} where T<:AbstractFloat
	# Order by objective value
	key = sortperm(objectives)
	obj_sorted = objectives[key]

	# Distance from values on either side
	shift_vec = [Inf; obj_sorted; Inf] # Edges have infinite distance
	prev_dist = obj_sorted - shift_vec[1:end-2] .|> abs
	next_dist = obj_sorted - shift_vec[3:end] .|> abs
	crowd = prev_dist + next_dist
	if obj_sorted[end] > obj_sorted[1]
		crowd *= abs(1 / obj_sorted[end-1] - obj_sorted[1]) # Normalize by fitness range
	end

	# Restore original order
	dist = Vector{T}(undef, length(key))
	dist[key] = crowd[:]

	return dist
end

function non_dominated_sort(objectives::Matrix{T})::Vector{T} where T <: Number
	# Non-dominated Sort
	fronts = get_fronts(objectives)
	# Rank each front by crowding distance
	for f in 1:length(fronts)
		x1 = objectives[fronts[f], 1]
		x2 = objectives[fronts[f], 2]
		crowdDist = get_crowding_dist(x1) + get_crowding_dist(x2)
		frontRank = sortperm(-crowdDist)
		fronts[f] = [fronts[f][i] for i in frontRank]
	end

	# Convert to ranking
	tmp = [ind for front in fronts for ind in front]
	rank = Vector{T}(undef, length(tmp))
	rank[tmp] = collect(1:length(tmp))

	return rank
end

if abspath(PROGRAM_FILE) == @__FILE__
	function main()
		# for test
		# get_random_index(f::Function, v::Matrix{<:AbstractFloat})::CartesianIndex{2} =
		# 	invoke((f, v) -> CartesianIndex(1, 2), Tuple{Function, Matrix{<:AbstractFloat}}, f, v)

		# v = [
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# 	0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		# ]
		v = [
			0.0 0.0 0.0 0.0 0.0 0.0;
			0.0 0.0 0.0 0.0 0.0 0.0;
			0.0 0.0 0.0 0.0 0.0 0.0;
			0.0 0.0 0.0 0.0 0.0 0.0;
			0.0 0.0 0.0 0.0 0.0 0.0;
			0.0 0.0 0.0 0.0 0.0 0.0;
		]
		a = [ActOrig(1) for _ in 1:6]
		# v = mutate_addconn(v, 1 + 1, 1, 2)
		# v = mutate_addnode(v, 1 + 1)
		for i = 0:0
			for _ in ([1:9, 1:5, 1:6])[i+1]
				v, a = mutate_addconn(v, a, 3, i, 3)
				println_matrix(v)
			end
			v, a = mutate_addnode(v, a, 3)
			println_matrix(v)
		end
		# v = mutate_addnode(v, 3)
		# println_matrix(v)
	end

	main()
end
