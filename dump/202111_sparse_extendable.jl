using SparseArrays, ExtendableSparse
using BenchmarkTools
using Test
import StatsBase: sample


##

function add_elements1!(mat::SparseMatrixCSC,ijs::Vector{CartesianIndex{2}})
  for ij in ijs
    mat[ij] = 123.456
  end
  return nothing
end

function add_elements2!(mat::SparseMatrixCSC,ijs::Vector{CartesianIndex{2}})
  mate = ExtendableSparseMatrix(mat)
  for ij in ijs
    i,j = Tuple(ij)
    setindex!(mate,123.456,i,j)
  end
  flush!(mate)
  copy!(mat,mate.cscmatrix)
  return nothing
end

##
# test that they do the same


N = 6000
mat = sprand(N,N,0.2)
nchange = 500
tochange = sample(CartesianIndices(mat),nchange;replace=false)
filter!(ij-> mat[ij]==0.0,tochange)
mat1 = deepcopy(mat)
mat2 = deepcopy(mat)

add_elements1!(mat1,tochange)
add_elements2!(mat2,tochange)

@test all( mat1 .== mat2)

##
@benchmark add_elements1!(_mat,elements) setup=(_mat=copy(mat);elements=tochange) 
##
@benchmark add_elements2!(_mat,elements) setup=(_mat=copy(mat);elements=tochange) 

##

function check_element2(mat::SparseMatrixCSC,ij::CartesianIndex{2})
  return ! iszero(mat[ij])
end

function check_element1(mat::SparseMatrixCSC,ij::CartesianIndex{2})
  i,j = Tuple(ij)
  return ! iszero(ExtendableSparse.findindex(mat,i,j))
end


@benchmark check_element1($mat,CartesianIndex(333,444))
@benchmark check_element2($mat,CartesianIndex(333,444))