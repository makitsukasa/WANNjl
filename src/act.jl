# activation function
abstract type Act end

function call(act::Act, x::Number)
	throw(error("not impremented"))
end

function call(act::Act, array::Vector{<:Number})
	return map(x -> call(act, x), array)
end

mutable struct ActOrig <: Act
	id::Int
end

ActOrig() = ActOrig(rand(1:10))

function call(act::ActOrig, x::Number)
	if act.id == 2      # Unsigned Step Function
		return x > 0.0 ? 1.0 : 0.0
		#return (tanh(50*x/2.0) + 1.0)/2.0
	elseif act.id == 3  # Sin
		return sin(MathConstants.pi * x)
	elseif act.id == 4  # Gaussian with mean 0 and sigma 1
		return exp(-(x^2) / 2.0)
	elseif act.id == 5  # Hyperbolic Tangent (signed)
		return tanh(x)
	elseif act.id == 6  # Sigmoid (unsigned)
		return (tanh(x / 2.0) + 1.0) / 2.0
	elseif act.id == 7  # Inverse
		return -x
	elseif act.id == 8  # Absolute Value
		return abs(x)
	elseif act.id == 9  # Relu
		return max(0.0, x)
	elseif act.id == 10 # Cosine
		return cos(MathConstants.pi * x)
	# elseif act.id == 11 # Squared
	# 	return x^2
	else                # Linear
		return x
	end
end

function mutate!(act::ActOrig)
	candidate = collect(1:10)
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
