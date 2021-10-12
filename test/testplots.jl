push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using ProgressMeter

function onesparsemat(w::Real)
  return sparse(fill(w,(1,1)))
end

using Random; Random.seed!(0)


##