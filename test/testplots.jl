push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
using Test
pkg"activate ."

using LinearAlgebra,Statistics,StatsBase
using Plots ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs


##
ne = 1000
ni = 150

τe = 20E-3
τi = 10E-3

pope = S.PopRateQuadratic(ne,τe,0.05)
popi = S.PopRateQuadratic(ni,τi,0.05)

pse = S.PSRateQuadratic(pope)
psi = S.PSRateQuadratic(popi)


ptest=0.01
μtest = 5.0
σtest = 3.0
wmatie = S.sparse_wmat_lognorm(popi,pope,ptest,μtest,σtest;noself=false,exact=true) 
wmatei = S.sparse_wmat_lognorm(pope,popi,ptest,-μtest,σtest;noself=false,exact=true) 

conn_ie = S.ConnectionRate(pse,wmatie,psi)
##

using BenchmarkTools



@benchmark S.send_signal($conn_ie)

mtest = randn(ni,ne)
uffi = zeros(Float64,ne)
bau = zeros(Float64,ni)

foo = function ()
  bau .+= mtest * uffi
end

@benchmark foo()