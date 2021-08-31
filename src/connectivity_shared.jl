# buidling connection matrices

# Utility functions
# for connectivity

"""
        lognorm_reparametrize(m,std) -> distr::LogNormal
# parameters
  + `m`   sample mean
  + `std` sample std
"""
function lognorm_reparametrize(m::Real,std::Real)
    vm2= (std/m)^2
    μ = log(m / sqrt(1. + vm2))
    σ = sqrt(log( 1. + vm2))
    return LogNormal(μ,σ)
end

function no_autapses!(mat::SparseMatrixCSC)
  for i in 1: min(size(mat)...)
    mat[i,i]=0.0
  end
  return dropzeros!(mat)
end

# scale rows so that the sum of the nonzero elements is exactly x
# for simplicity ignores the sign of x
function scale_rows!(x::Real,mat::SparseMatrixCSC) 
  x=abs(x) # there are better ways to change sign!
  n = size(mat,1)
  vals=nonzeros(mat)
  rows=rowvals(mat)
  for i in 1:n
    idxs = findall(r->r==i,rows)
    if !isempty(idxs) # is it control useful? or less efficient?
      s = sum(vals[idxs])
      vals[idxs] .*= x/s
    end
  end
  return mat
end

# scal is a product at the end, for example set to (-1) for inhibitory neurons,
# or 1/sqrt(N) etc
function _sparse_wmat(npost::Integer,npre::Integer,p::Real,make_weights!::Function ; 
    scal::Float64=1.0, # final scaling
    noself::Bool=true, # no autapses ?
    rowsum::Union{Nothing,Float64}=nothing) #mean of each row (before scal)
  wmat = sprand(Float64,npost,npre,p)
  if noself
    no_autapses!(wmat)
  end
  make_weights!(wmat)
  if !isnothing(rowsum) # scale so that each row has mean exactly rowsum mean
    scale_rows!(rowsum,wmat)
  end
  if scal != 1.0
    lmul!(scal,wmat)
  end
  return wmat
end

function sparse_wmat_lognorm(npost::Integer,npre::Integer,
    p::Real,μ::Real,σ::Real;
    noself::Bool=true, # no autapses ?
    rowsum::Union{Nothing,Float64}=nothing)
  μp = abs(μ)
  distr = lognorm_reparametrize(μp,σ)
  make_weights=function (smat)
    vals=nonzeros(smat)
    Distributions.rand!(distr,vals)
  end
  scal = μ < 0 ? -1.0 : 1.0
  return _sparse_wmat(npost,npre,p,make_weights;
    scal=scal,noself=noself,rowsum=rowsum)
end

function sparse_wmat(npost::Integer,npre::Integer,p::Real,j_val::Real ; 
    scal::Float64=1.0, # final scaling
    noself::Bool=true, # no autapses ?
    rowsum::Union{Nothing,Float64}=nothing)
  make_weights=function (smat)
    vals=nonzeros(smat)
    fill!(vals,j_val)
  end
  return _sparse_wmat(npost,npre,p,make_weights;
    scal=scal,noself=noself,rowsum=rowsum)
end



