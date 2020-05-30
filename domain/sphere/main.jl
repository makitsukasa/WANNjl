include("../../src/wann.jl")
using DataFrames # select
using CSV # read
using Statistics: mean

n_pop = 1000
n_generation = 1000

function reward(output, ans)
	return -mean(abs.(ans .- output) ./ ans)
end

function test(outputs, ans)
	n_test = size(ans, 1)
	diffs = []
	for o in outputs
		push!(diffs, mean(abs.(ans .- o) ./ ans))
	end
	println("avg diff : ", mean(diffs))
end

dataframe = CSV.read("domain/sphere/data/sphere5.csv", header=true, delim=",")
n_sample = size(dataframe)[1]
n_test = div(n_sample, 5)
in = convert(Matrix, select(dataframe, r"i"))
ans = convert(Matrix, select(dataframe, r"o"))
test_in = in[end-n_test+1:end, 1:end]
test_ans = ans[end-n_test+1:end, 1:end]
in = in[1:end-n_test, 1:end]
ans = ans[1:end-n_test, 1:end]

hyp = Dict(
	"select_cull_ratio" => 0.2,
	"select_elite_ratio"=> 0.2,
	"select_tourn_size" => 32,
	"prob_initEnable" => 0.05,
	"alg_probMoo" => 0.8,
	"prob_crossover" => 0.0
)

param_for_train = Dict(
	"pop" => WANN.Pop(size(in, 2), size(ans, 2), n_pop, hyp["prob_initEnable"]),
	"data" => in,
	"ans" => ans,
	"test_data" => test_in,
	"test_ans" => test_ans,
	"n_generation" => n_generation,
	"reward" => reward,
	"test" => test,
	"hyp" => hyp,
)

println("train")
WANN.train(param_for_train)
