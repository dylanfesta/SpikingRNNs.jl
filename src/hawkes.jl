# Hawkes point processes
# I will try to do exact simulations.


struct PSHawkes{P} <: PopulationState
  population::P
  state_now::Vector{Float64}
  spike_proposal::Vector{Float64}
end
function PSHawkes(p::Population)
  PSHawkes(p, [zeros(Float64,p.n) for _ in (1:4) ]... )
end


struct ConnectionHawkes <: Connection
  postps::PSHawkes # post population 
  preps::PSHawkes # pre population 
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
end
function ConnectionRate(post::PSHawkes,weights::SparseMatrixCSC,pre::PSHawkes)
  aR,aC,_ = findnz(weights)
  ConnectionRate(post,pre,aR,aC,weights)
end


function dynamics_step!(t::Real,dt::Float64,ps::PSHawkes)
  next_idx = argmin( s -> hawkes_next_spike(ps.population,s) ,state_now)

  return nothing
end
#=

# threshold-linear input-output function
struct PopRateReLU <: Population
  n::Int64 # pop size
  τ::Float64 # time constant
  α::Float64 # gain modulation
end
@inline function iofunction(x::Float64,p::PopRateReLU)
  if x<=0.0
    return 0.0
  else
    return x*p.α
  end
end
@inline function ioinv(y::Float64,p::PopRateReLU)
  y<=0.0 && return 0.0
  return y/p.α
end

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
@inline function ioinv(y::Float64,p::PopRateQuadratic)
  y<=0.0 && return 0.0
  return sqrt(y/p.α)
end


"""
    send_signal(conn::ConnectionStateRate)

Computes the input to postsynaptic population, given the current state of presynaptic population.
For a rate model, it applies the iofunction to the neuron potentials, gets the rate values
then multiplies rates by weights, adding the result to the input of the postsynaptic population.
"""
function send_signal!(t::Real,conn::ConnectionRate)
  # convert a state to rates r = iofun(u) 
  r = conn.preps.alloc_r
  broadcast!(x->iofunction(x,conn.preps),
    r,conn.preps.state_now)
  # multiply by weights, add to input of b .  input_b += W * r
  mul!(conn.postps.input,conn.weights,r,1,1)
  return nothing
end

# No plasticity here, or evolution of any kind
@inline function dynamics_step!(t_now::Real,dt::Real,conn::ConnectionRate)
  return nothing
end
#=