push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end


##
# isolated, non-interacting , processes 
# with given input