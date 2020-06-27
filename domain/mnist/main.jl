include("../../src/algorithm.jl")
include("../../src/wann.jl")
using LinearAlgebra: transpose!
using Flux: onehot
using CSV # read
using Statistics: mean

n_sample = 1000
n_test = 100
n_pop = 960
# n_generation = 4096
n_generation = 70
image_size = 16
file_name = "a.txt"

function reward(output, labels)
	# softmax cross entropy
	softmax = mapslices(x -> exp.(x) ./ sum(exp.(x)), output, dims = 2)
	if isnan(sum(labels .* log.(softmax .+ eps())))
		output_bigfloat = convert(Matrix{BigFloat}, output)
		softmax_bigfloat = mapslices(x -> exp.(x) ./ sum(exp.(x)), output_bigfloat, dims = 2)
		if isnan(sum(labels .* log.(softmax_bigfloat .+ eps())))
			println("NaN")
			println(typeof(sum(labels .* log.(softmax_bigfloat .+ eps())) / n_sample))
			open("hoge.txt", "a") do fp
				write(fp, "reward is NaN\n")
				write(fp, "output\n")
				println_matrix(fp, output)
				write(fp, "softmax\n")
				println_matrix(fp, softmax_bigfloat)
			end
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

	println("acculacy late : $(correct / (correct + incorrect))")
	return correct / (correct + incorrect)
end

imgdata = CSV.read("domain/mnist/data/train_img.csv", header=false, delim=" ")
labeldata = CSV.read("domain/mnist/data/train_label.csv", header=false, delim=" ")
if n_sample + n_test > size(imgdata)[1]
	println("n_sample + n_test > 60000")
	exit()
end
imgs = convert(Matrix, imgdata)
labels_onehot = map(x -> onehot(x, 0:9), convert(Matrix, labeldata))
labels = hcat([[labels_onehot[y][x] ? 1.0 : 0.0 for y = 1:60000] for x = 1:10]...)
test_imgs = imgs[end - n_test:end, :]
test_labels = labels[end - n_test:end, :]
imgs = imgs[1:n_sample, :]
labels = labels[1:n_sample, :]

for i in 1:1
	# prob = [rand() rand() rand() 0]
	# prob = [p / sum(prob) for p in prob]

	# # prob_2 = rand() * 0.15
	# # prob = [0.85 prob_2 (0.15 - prob_2) 0]

	# prob[4] = prob[1] * rand() * rand()
	# prob[1] = 0
	# prob[1] = 1 - sum(prob)

	# prob = [0.353 0.105 0.454 0.088]
	prob = [0.2 0.25 0.5 0.05]

	hyp = Dict(
		"select_cull_ratio" => 0.2,
		"select_elite_ratio"=> 0.2,
		"select_tourn_size" => 32,
		"prob_initEnable" => 0.05,
		"alg_probMoo" => 0.8,
		"prob_crossover" => 0.0,
		"prob_addnode" => prob[1],
		"prob_reviveconn" => prob[4],
		"prob_addconn" => prob[2],
		"prob_mutateact" => prob[3],
	)

	println("n_generation: $n_generation, prob_addnode:$(hyp["prob_addnode"]), prob_reviveconn:$(hyp["prob_reviveconn"]), prob_addconn:$(hyp["prob_addconn"]), prob_mutateact:$(hyp["prob_mutateact"])")

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
	ind = WANN.train(param_for_train)

	r = WANN.calc_rewards(ind, reward, imgs, labels)
	ac_rate = test([WANN.calc_output(ind, test_imgs, w) for w in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]], test_labels)

	open(file_name, "a") do fp
		write(fp, "ac_rate:$ac_rate,reward:$(mean(r)),prob_addnode:$(hyp["prob_addnode"]), prob_reviveconn:$(hyp["prob_reviveconn"]), prob_addconn:$(hyp["prob_addconn"]), prob_mutateact:$(hyp["prob_mutateact"])\n")
	end
	println("ac_rate:$ac_rate,reward:$(mean(r)),prob_addnode:$(hyp["prob_addnode"]), prob_reviveconn:$(hyp["prob_reviveconn"]), prob_addconn:$(hyp["prob_addconn"]), prob_mutateact:$(hyp["prob_mutateact"])")
end
