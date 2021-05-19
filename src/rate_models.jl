
struct PSRate{P} <: PopulationState
  population::P
  current_state::Vector{Float64}
  alloc_du::Vector{Float64}
  alloc_r::Vector{Float64}
  input::Vector{Float64}
end
function PSRate(p::Population)
  PSRate(p, [zeros(Float64,p.n) for _ in (1:4) ]... )
end

iofunction(x,ps::PSRate) = iofunction(x,ps.population)
ioinv(x,ps::PSRate) =ioinv(x,ps.population)


struct ConnectionRate <: Connection
  postps::PSRate # postsynaptic population state
  preps::PSRate # presynaptic population state
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
end
function ConnectionRate(post::PSRate,weights::SparseMatrixCSC,pre::PSRate)
  aR,aC,_ = findnz(weights)
  ConnectionRate(post,pre,aR,aC,weights)
end


function dynamics_step!(dt::Float64,ps::PSRate)
  copy!(ps.alloc_du,ps.current_state)  # u
  ps.alloc_du .-= ps.input  # (u-I)
  lmul!(-dt/ps.population.τ,ps.alloc_du) # du =  dt/τ (-u+I)
  ps.current_state .+= ps.alloc_du # u_t+1 = u_t + du
  return nothing
end


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
function send_signal!(conn::ConnectionRate)
  # convert a state to rates r = iofun(u) 
  r = conn.preps.alloc_r
  broadcast!(x->iofunction(x,conn.preps),
    r,conn.preps.current_state)
  # multiply by weights, add to input of b .  input_b += W * r
  mul!(conn.postps.input,conn.weights,r,1,1)
  return nothing
end
