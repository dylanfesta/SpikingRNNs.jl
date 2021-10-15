using SparseArrays
using BenchmarkTools
using Test
import StatsBase: sample
##

function sum_over_cols1!(dest::Vector{Float64},M::SparseMatrixCSC)
  _colptr = SparseArrays.getcolptr(M) # column indexing
  Mnz = nonzeros(M) # direct access to weights 
  @inbounds @simd for i in 1:length(_colptr)-1
    accu = 0.0
    for k in _colptr[i]:_colptr[i+1]-1
      accu += Mnz[k]
    end
    dest[i] = accu
  end
  return dest
end
function sum_over_cols1B!(dest::Vector{Float64},M::SparseMatrixCSC)
  @assert length(dest) == size(M,2)
  fill!(dest,0.0)
  _colptr = SparseArrays.getcolptr(M) # column indexing
  Mnz = nonzeros(M) # direct access to weights 
  @inbounds @simd for i in 1:length(_colptr)-1
    dest[i] += sum(Mnz[_colptr[i]:_colptr[i+1]-1] )
  end
  return dest
end

function sum_over_cols2!(dest::Matrix{Float64},M::SparseMatrixCSC)
  return sum!(dest,M)
end

function sum_over_cols3!(M::SparseMatrixCSC)
  return sum(M;dims=1)
end

function sum_over_cols4!(M::Matrix{Float64})
  return sum(M;dims=1)
end
##

const n = 6000
const spars = 0.3
function domat(_n,_spars)
  return sprand(_n,_n,_spars)
end

## first, test that functions return same vals
Mtest = domat(n,spars)
test1 = let ret = fill(NaN,n)
  sum_over_cols1!(ret,Mtest)
  ret
end
test1B = let ret = fill(NaN,n)
  sum_over_cols1B!(ret,Mtest)
  ret
end
test2 = let ret=fill(NaN,(1,n))
  sum_over_cols2!(ret,Mtest)
  ret[:]
end
test3 = sum_over_cols3!(Mtest)[:]
test4 = sum_over_cols4!(Matrix(Mtest))[:]

@test all(isapprox.(test1,test2))
@test all(isapprox.(test1,test1B))
@test all(isapprox.(test1,test3))
@test all(isapprox.(test1,test4))
# to visualize :
@show extrema(test1 .- test3)

## Now, benchmark them!
@info "Benchmark for function 1"
@benchmark sum_over_cols1!(dest,M) setup=(M=domat(n,spars);dest=Vector{Float64}(undef,n))
##
@info "Benchmark for function 1B"
@benchmark sum_over_cols1B!(dest,M) setup=(M=domat(n,spars);dest=Vector{Float64}(undef,n))
##
@info "Benchmark for function 2"
@benchmark sum_over_cols2!(dest,M) setup=(M=domat(n,spars);dest=Matrix{Float64}(undef,1,n))
##
@info "Benchmark for function 3"
@benchmark sum_over_cols3!(M) setup=(M=domat(n,spars))
##
@info "Benchmark for function 4"
@benchmark sum_over_cols4!(M) setup=(M=Matrix(domat(n,spars)))

##
########################
# Now repeat for row sum

function sum_over_rows1!(dest::Vector{Float64},M::SparseMatrixCSC)
  return sum!(dest,M)
end
function sum_over_rows2!(dest::Matrix{Float64},M::SparseMatrixCSC)
  Mtr = permutedims(M)
  return sum!(dest,Mtr)
end
function sum_over_rows3!(dest::Matrix{Float64},M::SparseMatrixCSC)
  Mtr = transpose(M)
  return sum!(dest,Mtr)
end
function sum_over_rows4!(dest::Vector{Float64},M::SparseMatrixCSC)
  fill!(dest,0.0)
  _rowidx = SparseArrays.rowvals(M)
  Mnz = nonzeros(M) # direct access to weights 
  @inbounds for (i,r) in enumerate(_rowidx)
    dest[r] += Mnz[i]
  end
  return dest
end
function sum_over_rows5!(M::SparseMatrixCSC)
  dest=zeros(size(M,1))
  _rowidx = SparseArrays.rowvals(M)
  Mnz = nonzeros(M) # direct access to weights 
  @inbounds for (i,r) in enumerate(_rowidx)
    dest[r] += Mnz[i]
  end
  return dest
end


##

test1 = let ret = fill(NaN,n)
  sum_over_rows1!(ret,Mtest)
  ret
end
test2 = let ret = fill(NaN,(1,n))
  sum_over_rows2!(ret,Mtest)
  ret[:]
end
test3 = let ret = fill(NaN,(1,n))
  sum_over_rows3!(ret,Mtest)
  ret[:]
end
test4 = let ret = fill(NaN,n)
  sum_over_rows4!(ret,Mtest)
  ret
end
test5 = sum_over_rows5!(Mtest)

@test all(isapprox.(test1,test2))
@test all(isapprox.(test1,test3))
@test all(isapprox.(test1,test4))
@test all(isapprox.(test1,test5))

##
@info "Benchmark for function 1"
@benchmark sum_over_rows1!(dest,M) setup=(M=domat(n,spars);dest=Vector{Float64}(undef,n))
##
@info "Benchmark for function 2"
@benchmark sum_over_rows2!(dest,M) setup=(M=domat(n,spars);dest=Matrix{Float64}(undef,(1,n)))
##
@info "Benchmark for function 3"
@benchmark sum_over_rows3!(dest,M) setup=(M=domat(n,spars);dest=Matrix{Float64}(undef,(1,n)))
##
@info "Benchmark for function 4"
@benchmark sum_over_rows4!(dest,M) setup=(M=domat(n,spars);dest=Vector{Float64}(undef,n))
##
@info "Benchmark for function 5"
@benchmark sum_over_rows5!(M) setup=(M=domat(n,spars))


##

const n = 6000
const n_sum = 400

function mysumloop(vec::Vector{Float64},idxs)
  ret=0.0
  @inbounds for idx in idxs
    ret += vec[idx]
  end
  return ret
end


##
@benchmark sum(view(vec,idxs)) setup=(vec=randn(n);idxs=sample(1:n,n_sum))
##
@benchmark mapreduce(idx->vec[idx],+,idxs) setup=(vec=randn(n);idxs=sample(1:n,n_sum))
##
@benchmark mysumloop(vec,idxs) setup=(vec=randn(n);idxs=sample(1:n,n_sum))
##
@benchmark sum(view(vec,idxs)) setup=(vec=randn(n);idxs=(1:n_sum))
##
@benchmark mapreduce(idx->vec[idx],+,idxs) setup=(vec=randn(n);idxs=(1:n_sum))
##
@benchmark mysumloop(vec,idxs) setup=(vec=randn(n);idxs=(1:n_sum))
