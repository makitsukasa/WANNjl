include("../../src/algorithm.jl")
include("../../src/wann.jl")
using LinearAlgebra: transpose!
using Flux: onehot
using Flux.Data.MNIST
using Images: imresize

n_sample = 1000
n_test_sample = 100
n_pop = 960
n_generation = 4096
image_size = 16

function reward(output, labels)
	# softmax cross entropy
	softmax = mapslices(x -> exp.(x) ./ sum(exp.(x)), output, dims = 2)
	if isnan(sum(labels .* log.(softmax .+ eps())))
		output_bigfloat = convert(Matrix{BigFloat}, output)
		softmax_bigfloat = mapslices(x -> exp.(x) ./ sum(exp.(x)), output_bigfloat, dims = 2)
		if isnan(sum(labels .* log.(softmax_bigfloat .+ eps())))
			println("NaN")
			println(typeof(sum(labels .* log.(softmax_bigfloat .+ eps())) / n_sample))kkkkkk
			exit()
		end
		return sum(labels .* log.(softmax_bigfloat .+ eps())) / n_sample
	end
	return sum(labels .* log.(softmax .+ eps())) / n_sample
end

function test(outputs, labels)
	correct = 0
	incorrect = 0
	for o in outputs
		for col in 1:size(o, 1)
			output_label = argmax(o[col, :])
			ans_label = argmax(labels[col, :])
			if output_label == ans_label
				correct += 1
			else
				incorrect += 1
			end
		end
	end
	println("acculacy late : ", correct / (correct + incorrect))
end

# convert Array{Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}},2},1} into Array{Float64,2}
imgs = zeros(Float64, (n_sample, image_size^2))
# imgs_f = convert(Vector{Matrix{Float64}}, MNIST.images(:train))
imgs_f = convert(Vector{Matrix{Float64}}, map(i -> imresize(i, (image_size, image_size)), MNIST.images(:train)))
transpose!(imgs, hcat(vec.(imgs_f)[1:n_sample, :]...))
# convert Array{Int64,1} into Array{Flux.OneHotVector,2}
hoge = map(x -> onehot(x, 0:9), MNIST.labels(:train)[1:n_sample, :])
labels = hcat([[hoge[y][x] ? 1.0 : 0.0 for y = 1:n_sample] for x = 1:10]...)

# same for test
test_imgs = zeros(Float64, (n_test_sample, image_size^2))
test_imgs_f = convert(Vector{Matrix{Float64}}, map(i -> imresize(i, (image_size, image_size)), MNIST.images(:test)))
transpose!(test_imgs, hcat(vec.(test_imgs_f)[1:n_test_sample, :]...))
hoge = map(x -> onehot(x, 0:9), MNIST.labels(:test)[1:n_test_sample, :])
test_labels = hcat([[hoge[y][x] ? 1.0 : 0.0 for y = 1:n_test_sample] for x = 1:10]...)

hyp = Dict(
	"select_cull_ratio" => 0.2,
	"select_elite_ratio"=> 0.2,
	"select_tourn_size" => 32,
	"prob_initEnable" => 0.05,
	"alg_probMoo" => 0.8,
	"prob_crossover" => 0.0
)

param_for_train = Dict(
	"pop" => WANN.Pop(image_size^2, 10, n_pop, hyp["prob_initEnable"]),
	"data" => imgs,
	"ans" => labels,
	"test_data" => test_imgs,
	"test_ans" => test_labels,
	"n_generation" => n_generation,
	"reward" => reward,
	"test" => test,
	"hyp" => hyp,
)

println("train")
WANN.train(param_for_train)
