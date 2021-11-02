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

function _sparse_wmat(npost::Integer,npre::Integer,p::Real,make_weights!::Function ; 
    scal::Float64=1.0, # final scaling (e.g. -1 for inhibitory neurons)
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

function sparse_wmat_rowfun(make_a_row::Function,
    npost::Integer,npre::Integer,p::Real;no_autapses::Bool=false)
  wmat = sprand(Float64,npost,npre,p)
  if no_autapses
    no_autapses!(wmat)
  end
  rw = rowvals(wmat)
  wvec = nonzeros(wmat)
  for r in unique(rw)
    ridx = r .== rw
    n = count(ridx)
    newrow = make_a_row(n)
    wvec[ridx] .= newrow
  end
  return wmat
end

function sparse_constant_wmat(npost::Integer,npre::Integer,p::Real,j_val::Real ; 
    scal::Float64=1.0, # final scaling
    no_autapses::Bool=true, # no autapses ?
    rowsum::Union{Nothing,Float64}=nothing)
  make_weights=function(n::Integer)
    ret=fill(j_val,n)
    if !isnothing(rowsum)
      ret .*= rowsum / sum(ret)
    end
    if scal != 1
      lmul!(scal,ret)
    end
    return ret
  end
  return sparse_wmat_rowfun(make_weights,npost,npre,p;no_autapses=no_autapses)
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

# not needed with new Poisson input types!
# """
#     wmat_train_assemblies_protocol(Nneus::Integer,p::Real; scal::Float64=1.0) 
#       -> (weights::SparseMatrixCSC,assembly_dictionary::Dict)

#   # Arguments
#   + Npost::Integer : number of postsynaptic neurons
#   + Nassemblies::Integer : number of assemblies to train
#   + p_assembly::Real : probability for a neuron to be in any assembly
#   + scal::Real=1.0 : connection weights

#   # Outputs
#   + weights : sparse weight matrix
#   + assembly_list::Vector{NamedTuple{(:neupost,:neupre),Vector{Int64,Vector{Int64}}}}
#     element `i` contains the pre and post neurons associated to assembly `i`
# """
# function wmat_train_assemblies_protocol(Npost::Integer,Nassemblies::Integer,
#     p_assembly::Real;scal::Float64=1.0)
#   # unoptimized... but called only once
#   post_neus = [findall(rand(Npost) .< p_assembly) for _ in 1:Nassemblies]
#   c = 0
#   pre_neus = map(post_neus) do _po
#     _ln = length(_po)
#     ret = c .+ collect(1:_ln)
#     c += _ln
#     return ret
#   end
#   _pre_neus_all = vcat(pre_neus...)
#   wmat = sparse(vcat(post_neus...),_pre_neus_all,scal,
#     Npost,length(_pre_neus_all)) # include total matrix size
#   assembly_list = map(zip(post_neus,pre_neus)) do (_post,_pre)
#     (neupost=_post,neupre=_pre)
#   end
#   return wmat,assembly_list
# end


# to order assemblies (no idea where to place this function)
function order_by_pattern_idxs(pattern_idxs::Vector{Vector{Int64}},nneus::Int64)
  _tosort = map(1:nneus) do neu
    pa = findall(patt->neu in patt,pattern_idxs)
    isnothing(pa) && return 10_000_000
    (length(pa) == 1) && return pa[1]
    (length(pa)>1) && return sample(pa)
  end
  return sortperm(_tosort;rev=true)
end
