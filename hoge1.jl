include("src/algorithm.jl")

for _ in 1:10

	v = Float64[
		0 1 2 3 4;
		0 0 0 6 7;
		0 0 0 0 8;
		0 0 0 0 9;
		0 0 0 0 0;
	]

	nIn = 1
	nOut = 1

	order = get_shuffued_order(v, nIn, nOut)
	println(order)
	order = get_sorted_order(v, nIn, nOut, order)
	println(order)
	get_assigned_v(v, order, Dict{CartesianIndex{2}, Float64}(), 0.0)
	# println_matrix(v)

	println()
end
