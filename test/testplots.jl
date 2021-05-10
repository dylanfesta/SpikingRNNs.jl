push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

## 
using SparseArrays

mtest = sprand(1000,100,0.01)

##
m=1000
n=100
nz=mtest.nzval
cc=mtest.colptr

rr=rowvals(mtest)

mtest2 = SparseMatrixCSC(m,n,cc,rr,nz)


all(mtest.==mtest2)