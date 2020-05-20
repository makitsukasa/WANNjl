include("./wann.jl")
using LinearAlgebra: transpose!
using Flux: onehot
using Flux.Data.MNIST

nSample = 100

# convert Array{Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}},2},1} into Array{Float64,2}
imgs = zeros(Float64, (nSample, 28^2))
transpose!(imgs, hcat(vec.(float.(MNIST.images()))[1:nSample, :]...))
# convert Array{Int64,1} into Array{Flux.OneHotVector,2}
hoge = map(x -> onehot(x, 0:9), MNIST.labels()[1:nSample, :])
labels = hcat([[hoge[y][x] ? 1.0 : 0.0 for y = 1:nSample] for x = 1:10]...)

println("typeof(imgs): ", typeof(imgs))
println("axes(imgs): ", axes(imgs))
println("typeof(labels): ", typeof(labels))
println("axes(labels): ", axes(labels))
println("")

hyp = Dict(
	"select_cull_ratio" => 0.2,
	"select_elite_ratio"=> 0.2,
	"select_tourn_size" => 32,
	"prob_crossover" => 0.0
)

pop = WANN.Pop(28^2, 100, nSample)
WANN.train(pop, imgs, labels, 10, hyp)

# in = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0; 0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0;]
# ans = [0.0 0.0 0.0 1.0; 0.0 1.0 1.0 1.0; 0.0 1.0 1.0 0.0; 1.0 0.0 0.0 1.0;
#        0.0 0.0 0.0 1.0; 0.0 1.0 1.0 1.0; 0.0 1.0 1.0 0.0; 1.0 0.0 0.0 1.0;]
# pop = WANN.Pop(2, 4, 4)
# WANN.train(pop, in, ans, 10)
