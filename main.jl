include("./wann.jl")
using LinearAlgebra: transpose!
using Flux: onehot
using Flux.Data.MNIST
using Images: imresize

n_sample = 1000
n_sample_test = 100
n_pop = 960
n_generation = 100
image_size = 16

# convert Array{Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}},2},1} into Array{Float64,2}
imgs = zeros(Float64, (n_sample, image_size^2))
# imgs_f = convert(Vector{Matrix{Float64}}, MNIST.images(:train))
imgs_f = convert(Vector{Matrix{Float64}}, map(i -> imresize(i, (image_size, image_size)), MNIST.images(:train)))
transpose!(imgs, hcat(vec.(imgs_f)[1:n_sample, :]...))
# convert Array{Int64,1} into Array{Flux.OneHotVector,2}
hoge = map(x -> onehot(x, 0:9), MNIST.labels(:train)[1:n_sample, :])
labels = hcat([[hoge[y][x] ? 1.0 : 0.0 for y = 1:n_sample] for x = 1:10]...)

# convert Array{Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}},2},1} into Array{Float64,2}
imgs_test = zeros(Float64, (n_sample_test, image_size^2))
# imgs_test_f = convert(Vector{Matrix{Float64}}, MNIST.images(:test))
imgs_test_f = convert(Vector{Matrix{Float64}}, map(i -> imresize(i, (image_size, image_size)), MNIST.images(:test)))
transpose!(imgs_test, hcat(vec.(imgs_test_f)[1:n_sample_test, :]...))
# convert Array{Int64,1} into Array{Flux.OneHotVector,2}
hoge = map(x -> onehot(x, 0:9), MNIST.labels(:test)[1:n_sample_test, :])
labels_test = hcat([[hoge[y][x] ? 1.0 : 0.0 for y = 1:n_sample_test] for x = 1:10]...)

# println("typeof(imgs): ", typeof(imgs))
# println("axes(imgs): ", axes(imgs))
# println("typeof(labels): ", typeof(labels))
# println("axes(labels): ", axes(labels))
# println("labels: ", labels)
# println("")

hyp = Dict(
	"select_cull_ratio" => 0.2,
	"select_elite_ratio"=> 0.2,
	"select_tourn_size" => 32,
	"prob_initEnable" => 0.05,
	"alg_probMoo" => 0.8,
	"prob_crossover" => 0.0
)

pop = WANN.Pop(image_size^2, 10, n_pop, hyp["prob_initEnable"])
println("train")
WANN.train(pop, imgs, labels, imgs_test, labels_test, n_generation, hyp)

# in = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0; 0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0;]
# ans = [0.0 0.0 0.0 1.0; 0.0 1.0 1.0 1.0; 0.0 1.0 1.0 0.0; 1.0 0.0 0.0 1.0;
#        0.0 0.0 0.0 1.0; 0.0 1.0 1.0 1.0; 0.0 1.0 1.0 0.0; 1.0 0.0 0.0 1.0;]
# pop = WANN.Pop(2, 4, 4)
# WANN.train(pop, in, ans, 10)
