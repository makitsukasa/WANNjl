# activation function
abstract type Act end

function get_function(act::Act)
	throw(error("not impremented"))
end

function call(act::Act, x::AbstractFloat)
	return get_function(act)(x)
end

function call(act::Act, x::Array{<:AbstractFloat})
	return map(get_function(act), x)
end

function call(acts::Array{<:Act}, x::Array{<:AbstractFloat})
	ans = zeros(size(x))
	println(size(acts), ", ", size(x))
	for index in eachindex(ans)
		ans[index] = call(acts[index], x[index])
	end
	return ans
end


mutable struct ActOrig <: Act
	id::Int
end

ActOrig() = ActOrig(rand(1:11))

function get_function(act::ActOrig)
	if act.id == 2      # Unsigned Step Function
		return x -> x > 0.0 ? 1.0 : 0.0
		#return (tanh(50*x/2.0) + 1.0)/2.0
	elseif act.id == 3  # Sin
		return x -> sin(MathConstants.pi * x)
	elseif act.id == 4  # Gaussian with mean 0 and sigma 1
		return x -> exp(-(x^2) / 2.0)
	elseif act.id == 5  # Hyperbolic Tangent (signed)
		return x -> tanh(x)
	elseif act.id == 6  # Sigmoid (unsigned)
		return x -> (tanh(x / 2.0) + 1.0) / 2.0
	elseif act.id == 7  # Inverse
		return x -> -x
	elseif act.id == 8  # Absolute Value
		return x -> abs(x)
	elseif act.id == 9  # Relu
		return x -> max(0.0, x)
	elseif act.id == 10 # Cosine
		return x -> cos(MathConstants.pi * x)
	elseif act.id == 11 # Squared
		return x -> x^2
	else                # Linear
		return x -> x
	end
end

function mutate!(act::ActOrig)
	candidate = collect(1:11)
	deleteat!(candidate, act.id)
	act.id = rand(candidate)
end

if abspath(PROGRAM_FILE) == @__FILE__

	function main()
		for actid in 1:11
			for val in -1.0:0.1:1.01
				println(call(ActOrig(actid), val))
			end
			println()
		end
	end

	main()

end
