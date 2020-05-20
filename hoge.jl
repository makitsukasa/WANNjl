function getFronts(objectives)
	values1 = objectives[1:end,1]
	values2 = objectives[1:end,2]

	# println(values1)
	# println(values2)

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
	return front[1:end-1]
  end

  function get_crowding_dist(objectives::Vector{T}) where T <: AbstractFloat
	if length(objectives) <= 1
	  return objectives
	end
	  # Order by objective value
	  key = sortperm(objectives)
	  obj_sorted = objectives[key]

	  # Distance from values on either side
	  shift_vec = [Inf; obj_sorted; Inf] # Edges have infinite distance
	  prev_dist = obj_sorted - shift_vec[1:end-2] .|> abs
	  next_dist = obj_sorted - shift_vec[2:end-1] .|> abs
	  crowd = prev_dist + next_dist
	  if (obj_sorted[end-1] - obj_sorted[1]) > 0
		  crowd *= abs(1 / obj_sorted[end-1] - obj_sorted[1]) # Normalize by fitness range
	  end

	  # Restore original order
	  dist = Vector{T}(undef, length(key))
	  dist[key] = crowd[:]

	  return dist
  end

  a = [1.0 4.0 6.0]
  b = [3.0 1.0 1.0]
  obj = transpose([a; b])
  fronts = getFronts(obj)
  println(fronts)
  for f in 1:length(fronts)
	x1 = obj[fronts[f], 1]
	x2 = obj[fronts[f], 2]
	crowdDist = get_crowding_dist(x1) + get_crowding_dist(x2)
	frontRank = reverse(sortperm(crowdDist))
	fronts[f] = [fronts[f][i] for i in frontRank]
  end
  println(fronts)
