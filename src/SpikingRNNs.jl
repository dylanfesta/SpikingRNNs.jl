module SpikingRNNs
using LinearAlgebra,Statistics,StatsBase,Random,Distributions
using SparseArrays
using ConfParser



# Utility functions

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


function sparse_wmat_lognorm(npre::Integer,npost::Integer,
    p::Real,μ::Real,σ::Real;exact=true)
  wmat = sprand(Float64,npost,npre,p)
  no_autapses!(wmat)
  d = lognorm_reparametrize(μ,σ)
  vals=nonzeros(wmat)
  rand!(d,vals)
  if exact # scale so that each row has mean exactly μ
    scale_rows!(μ,wmat)
  end
  return wmat
end

#

abstract type Population end
abstract type Connection end
abstract type Input end

abstract type PopState end
abstract type ConnState end



# warm-up , write a continuous rate module
# solved with Euler

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
  curr_state::Vector{Float64}
  alloc_du::Vector{Float64}
  alloc_r::Vector{Float64}
  input::Vector{Float64}
end
function PSRateQuadratic(p::PopRateQuadratic)
  PSRateQuadratic(p.n, ( zeros(p.n) for _ in 1:4  )... )
end


struct ConnRate <: Connection
  p::Float64 # connection probability
  μ::Float64 # mean of connected neurons (lognomr)
  σ::Float64 # variance of connected neurons (lognorm)
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
end

struct ConnStateRate <: ConnState
  weights::SparseMatrixCSC{Float64,Int64}
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

function send_signal(psb::PSRateQuadratic,conn::ConnStateRate,psa::PSRateQuadratic)
  # convert a state to rates r = iofun(u) 
  broadcast!(x->iofunction(x,psa.population),psa.alloc_r,psa.current_state)
  # multiply by weights, add to input of b .  input_b += W * r
  vals = nonzeros(conn.weights)
  wr = rowvals(conn.weights)
  for j in 1:psa.population.n
    for ii in nzrange(conn.weights,j)
      psb.input[wr[ii]] += vals[ii]*psa.alloc_r[j]
    end
  end 
  return nothing
end


function send_signal_simple(psb::PSRateQuadratic,conn::ConnStateRate,psa::PSRateQuadratic)
  # convert a state to rates r = iofun(u) 
  broadcast!(x->iofunction(x,psa.population),psa.alloc_r,psa.current_state)
  # multiply by weights, add to input of b .  input_b += W * r
  BLAS.gemv!(`N`,1.0,conn.weights,psa.alloc_r,1.0,psb.input)
  #psb.input += conn.weights * psa.alloc_r
  return nothing
end

end # of SpikingRNNs module 