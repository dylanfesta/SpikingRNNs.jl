using SparseArrays
using BenchmarkTools

n = 10_000

mat = sprandn(n,n,0.333)

##

rowv = SparseArrays.rowvals(mat)
colptr = SparseArrays.getcolptr(mat)

function findi1(i::Integer,j::Integer,mat::SparseMatrixCSC)
  ret = mat[i,j]
  (ret!=0.0) && (return ret)
  return nothing
end

function findi2(i::Integer,j::Integer,mat::SparseMatrixCSC)
  rowv = SparseArrays.rowvals(mat)
  colptr = SparseArrays.getcolptr(mat)
  start = colptr[j]
  stop = colptr[j+1]-1
  ret = findfirst(==(i),view(rowv,start:stop))
  isnothing(ret) && (return ret)
  return start+ret-1
end

function findi3(i::Integer,j::Integer,mat::SparseMatrixCSC)
  rowv = SparseArrays.rowvals(mat)
  colptr = SparseArrays.getcolptr(mat)
  start = colptr[j]
  stop = colptr[j+1]-1
  ret = searchsorted(rowv,i,start,stop,Base.Order.Forward)
  isempty(ret) && (return nothing)
  return first(ret)
end

function findi4(i::Integer,j::Integer,mat::SparseMatrixCSC)
  rowv = SparseArrays.rowvals(mat)
  colptr = SparseArrays.getcolptr(mat)
  start = colptr[j]
  stop = colptr[j+1]-1
  if rowv[start] <= i <= rowv[stop] 
    idx_ret = searchsortedfirst(rowv,i,start,stop,Base.Order.Forward)
    (rowv[idx_ret] == i) && (return idx_ret)
  end
  return nothing
end

function findi4less(i::Integer,rowv,start,stop)
  idx_ret = searchsortedfirst(rowv,i,start,stop,Base.Order.Forward)
  (rowv[idx_ret] == i) && (return idx_ret)
  return nothing
end

##
@benchmark findi1(i,j,$mat) setup=(i=rand(1:n);j=rand(1:n))
##
#@benchmark findi2(i,j,$mat) setup=(i=rand(1:n);j=rand(1:n))
##
@benchmark findi3(i,j,$mat) setup=(i=rand(1:n);j=rand(1:n))
##
@benchmark findi4(i,j,$mat) setup=(i=rand(1:n);j=rand(1:n))
##
@benchmark findi4less(i,$rowv,start,stop) setup=(i=rand(1:n);j=rand(1:n);start=colptr[j];stop=colptr[j+1]-1)