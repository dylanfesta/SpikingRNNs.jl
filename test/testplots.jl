push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

## 
using LinearAlgebra,Statistics,StatsBase
using Plots ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs


##
m,n = (8_000,10_000)
ptest = 0.001
μtest = 3.0
σtest = 0.5
wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest)

histogram(nonzeros(wtest);xlims=(0,8))

sum(wtest;dims=2)

mean(nonzeros(wtest))
std(nonzeros(wtest))


histogram(sum(wtest;dims=2)[:] .- m*ptest*μtest)

mtest = sprand(Float64,1000,100,0.01)

##
rr=rowvals(wtest)


##
mat = sparse(2I,4,4)

mat[1,4] = 345

vals=nonzeros(mat)

rowvals(mat)

vals[rowvals(mat).==1] .= 999

m=1000



n=100
nz=mtest.nzval
cc=mtest.colptr

rr=rowvals(mtest)

mtest2 = SparseMatrixCSC(m,n,cc,rr,nz)


all(mtest.==mtest2)