
push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs

##

Ne = 4000
p_as = 5E-2

##

wmat,pattidx = SpikingRNNs.wmat_train_assemblies_protocol(Ne,5,p_as;scal=123.0)

patt_t,patt_idx=S._patterns_train_uniform(10,0.5,30.)

patt_mat = S.binary_patterns_mat(Ne,pattidx,0.0,1.0)

##
heatmap(patt_mat)