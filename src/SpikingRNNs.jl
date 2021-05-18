module SpikingRNNs
using LinearAlgebra,Statistics,StatsBase,Random,Distributions
using SparseArrays
using ConfParser

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


# Base elements

abstract type Population end
abstract type Connection end
abstract type Input end

abstract type PopState end
abstract type ConnState end


# Simplified form using abstract types
sparse_wmat_lognorm(poppost::Population,poppre::Population,p,μ,σ;noself=false,exact=true) = 
  sparse_wmat_lognorm(poppost.n,poppre.n,p,μ,σ;noself=noself,exact=exact)

# continuous rate module, supralinear activation function

# threshold-quadratic input-output function
struct PopRateQuadratic <: Population
  n::Int64 # pop size
  τ::Float64 # time constant
  α::Float64 # gain modulation
end
@inline function iofunction(x::Float64,p::PopRateQuadratic)
  if x<=0.0
    return 0.0
  else
    return x*x*p.α
  end
end

struct PSRateQuadratic <: PopState
  population::PopRateQuadratic
  current_state::Vector{Float64}
  alloc_du::Vector{Float64}
  alloc_r::Vector{Float64}
  input::Vector{Float64}
end
function PSRateQuadratic(p::PopRateQuadratic)
  PSRateQuadratic(p, [zeros(Float64,p.n) for _ in (1:4) ]... )
end

iofunction(x,ps::PSRateQuadratic) = iofunction(x,ps.population)


struct ConnectionRate <: Connection
  preps::PSRateQuadratic # presynaptic population state
  postps::PSRateQuadratic # postsynaptic population state
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
end
function ConnectionRate(pre::PSRateQuadratic,weights::SparseMatrixCSC,post::PSRateQuadratic)
  aR,aC,_ = findnz(weights)
  ConnectionRate(pre,post,aR,aC,weights)
end


struct InputRateStatic <: Input
  h::Vector{Float64}
end

function dynstep(dt::Float64,ps::PSRateQuadratic)
  copy!(ps.alloc_du,ps.current_state)  # u
  ps.alloc_du .-= ps.input  # (u-I)
  lmul!(-dt/ps.p.τ,ps.alloc_du) # du =  dt/τ (-u+I)
  ps.curr_state .+= ps.alloc_du # u_t+1 = u_t + du
  fill!(ps.input,0.0) # reset the input
  return nothing
end

"""
    send_signal(conn::ConnStateRate)

Computes the input to postsynaptic population, given the current state of presynaptic population.
For a rate model, it applies the iofunction to the neuron potentials, gets the rate values
then multiplies rates by weights, adding the result to the input of the postsynaptic population.
"""
function send_signal(conn::ConnectionRate)
  # convert a state to rates r = iofun(u) 
  r = conn.preps.alloc_r
  broadcast!(x->iofunction(x,conn.preps),
    r,conn.preps.current_state)
  # multiply by weights, add to input of b .  input_b += W * r
  muladd(conn.weights,r,conn.postps.input)
  return nothing
end



end # of SpikingRNNs module 