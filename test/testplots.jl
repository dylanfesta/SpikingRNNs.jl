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


hom_add=S.HeterosynapticAdditive(minval)
hom_mul=S.HeterosynapticMultiplicative(minval)
hom_targ = S.HeterosynapticOutgoing(400.0)

v_test = copy(v_test0)
v_test2 = copy(v_test0)
v_test3 = copy(v_test0)
idx_test = copy(idx_test0)
idx_test2 = copy(idx_test0)
idx_test3 = copy(idx_test0)
S._heterosynaptic_fix!(v_test,idx_test,hom_add,hom_targ)

S._heterosynaptic_fix!(v_test2,idx_test2,hom_mul,hom_targ)

S._heterosynaptic_fix2!(v_test3,idx_test3,hom_mul,hom_targ)


##

@benchmark S._heterosynaptic_fix!(v,idx,$hom_mul,$hom_targ) setup=(v=copy(v_test);idx=copy(idx_test))
##
@benchmark S._heterosynaptic_fix2!(v,idx,$hom_mul,$hom_targ) setup=(v=copy(v_test);idx=copy(idx_test))

##
sum(v_test[idx_test0])
sum(v_test2[idx_test0])
sum(v_test3[idx_test0])
sum(v_test0[idx_test0])

minimum(v_test[idx_test0])
minimum(v_test2[idx_test0])
minimum(v_test3[idx_test0])
minimum(v_test0[idx_test0])

##

wtest = sprand(8,5,0.6)

hom_targ_out = S.HeterosynapticOutgoing(400.0)
hom_targ_in = S.HeterosynapticIncoming(400.0)
it = S.HeterosynapticIdxsIterator(wtest,hom_targ_out)
it2 = S.HeterosynapticIdxsIterator(wtest,hom_targ_in)
##

_ = let u = 0
  for idxs in it2
    @show nonzeros(wtest)[idxs]
  end
end