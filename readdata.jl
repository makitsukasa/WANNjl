using DataFrames # select
using CSV # read

dataframe = CSV.read("data/sphere2.csv", header=true, delim=",")
# @show dataframe

@show select(dataframe, r"i")

input = select(dataframe, r"i")
