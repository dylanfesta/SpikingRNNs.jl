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
  n = min(size(mat)...)
  for i in 1:n
    mat[i,i]=0.0
  end
  return dropzeros!(mat)
end

# scale rows so that the sum of the nonzero elements is exactly x
function scale_rows!(x::Real,mat::SparseMatrixCSC) 
  n,_ = size(mat)
  vals=nonzeros(mat)
  rows=rowvals(mat)
  for i in 1:n
    idxs = findall(x->x==i,rows)
    s = sum(vals[idxs])
    vals[idxs] .*= x/s
  end
  return mat
end


function sparse_wmat_lognorm(npost::Integer,npre::Integer,
    p::Real,μ::Real,σ::Real;noself=false,exact=true)
  wmat = sprand(Float64,npost,npre,p)
  μp = abs(μ)
  if noself
    no_autapses!(wmat)
  end
  d = lognorm_reparametrize(μp,σ)
  vals=nonzeros(wmat)
  rand!(d,vals)
  if exact # scale so that each row has mean exactly μ
    scale_rows!(μp,wmat)
  end
  if μ < 0.0
    return - wmat
  else
    return wmat
  end
end


# Simplified form using abstract types
sparse_wmat_lognorm(poppost::Population,poppre::Population,p,μ,σ;noself=false,exact=true) = 
  sparse_wmat_lognorm(poppost.n,poppre.n,p,μ,σ;noself=noself,exact=exact)

