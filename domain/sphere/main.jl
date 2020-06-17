include("../../src/algorithm.jl")
include("../../src/wann.jl")
using DataFrames # select
using CSV # read
using Statistics: mean

n_pop = 1000
n_sample = 2^63 - 1
n_generation = 10000

function reward(output, ans)
	return -mean(abs.(ans .- output) ./ ans)
end

function test(outputs, ans)
	diffs = []
	for o in outputs
		push!(diffs, mean(abs.(ans .- o) ./ ans))
	end

	println("avg diff : $(mean(diffs)), min diff : $(minimum(diffs)), output : $(mean(outputs))")
end

dataframe = CSV.read("domain/sphere/data/sphere5.csv", header=true, delim=",")
n_sample = min(n_sample, div(size(dataframe)[1], 5) * 4)
n_test = max(1, div(n_sample, 5))
in = convert(Matrix, select(dataframe, r"i"))
ans = convert(Matrix, select(dataframe, r"o"))
test_in = in[(n_sample - n_test + 1):n_sample, :]
test_ans = ans[(n_sample - n_test + 1):n_sample, :]
in = in[1:(n_sample - n_test), :]
ans = ans[1:(n_sample - n_test), :]

hyp = Dict(
	"select_cull_ratio" => 0.0,
	"select_elite_ratio"=> 0.2,
	"select_tourn_size" => 32,
	"prob_initEnable" => 0.2,
	"alg_probMoo" => 0.8,
	"prob_crossover" => 0.0,
	"prob_addnode" => 0.2,
	"prob_reviveconn" => 0.05,
	"prob_addconn" => 0.45,
	"prob_mutateact" => 0.3,
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
ind = WANN.train(param_for_train)
r = calc_rewards(ind, reward, data, ans)

