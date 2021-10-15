push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using ProgressMeter

using InvertedIndices

function onesparsemat(w::Real)
  return sparse(fill(w,(1,1)))
end

using Random; Random.seed!(0)

##
const minval = 0.3
v_test0 =rand(Exponential(1.0),10_000) 
#v_test0 = max.(minval,v_test0)
idx_test0 = sample(1:10_000,800;replace=false)

##

ZZZZZZZZZZzzz

hom_add=S.HomeostaticAdditive(minval)
hom_mul=S.HomeostaticMultiplicative(minval)
hom_targ = S.HomeostaticOutgoing(400.0)

v_test = copy(v_test0)
idx_test = copy(idx_test0)
S._homeostatic_fix!(v_test,idx_test,hom_add,hom_targ)

v_test2 = copy(v_test0)
idx_test2 = copy(idx_test0)
S._homeostatic_fix!(v_test2,idx_test2,hom_mul,hom_targ)


##
sum(v_test[idx_test0])
sum(v_test2[idx_test0])
sum(v_test0[idx_test0])

minimum(v_test[idx_test0])
minimum(v_test2[idx_test0])
minimum(v_test0[idx_test0])

