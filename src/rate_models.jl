
struct PSRate{P} <: PopulationState{P}
  n::Int64 # population size
  neurontype::P
  state_now::Vector{Float64}
  alloc_du::Vector{Float64}
  alloc_r::Vector{Float64}
  input::Vector{Float64}
end
function PSRate(p::NeuronType,n::Integer)
  PSRate(n,p,[zeros(Float64,n) for _ in (1:4) ]... )
end

iofunction(x,ps::PSRate) = iofunction(x,ps.neurontype)
ioinv(x,ps::PSRate) =ioinv(x,ps.neurontype)


function local_update!(t::Real,dt::Float64,ps::PSRate)
  copy!(ps.alloc_du,ps.state_now)  # u
  ps.alloc_du .-= ps.input  # (u-I)
  lmul!(-dt/ps.neurontype.τ,ps.alloc_du) # du =  dt/τ (-u+I)
  ps.state_now .+= ps.alloc_du # u_t+1 = u_t + du
  return nothing
end


# threshold-linear input-output function
struct NTReLU <: NeuronType
  τ::Float64 # time constant
  α::Float64 # gain modulation
end
@inline function iofunction(x::Float64,p::NTReLU)
  if x<=0.0
    return 0.0
  else
    return x*p.α
  end
end
@inline function ioinv(y::Float64,p::NTReLU)
  y<=0.0 && return 0.0
  return y/p.α
end

# threshold-quadratic input-output function
struct NTQuadratic <: NeuronType
  τ::Float64 # time constant
  α::Float64 # gain modulation
end
@inline function iofunction(x::Float64,p::NTQuadratic)
  if x<=0.0
    return 0.0
  else
    return x*x*p.α
  end
end
@inline function ioinv(y::Float64,p::NTQuadratic)
  y<=0.0 && return 0.0
  return sqrt(y/p.α)
end

struct NTStep <: NeuronType
  θ::Float64 # step boundary
end
@inline function iofunction(x::Float64,p::NTStep)
  return (x<p.θ ? zero(Float64) : one(Float64)) 
end



"""
    send_signal(conn::ConnectionStateRate)

Computes the input to postsynaptic population, given the current state of presynaptic population.
For a rate model, it applies the iofunction to the neuron potentials, gets the rate values
then multiplies rates by weights, adding the result to the input of the postsynaptic population.
"""
function forward_signal!(t::Real,dt::Real,pspost::PSRate,conn::BaseConnection,pspre::PSRate)
  # convert a state to rates r = iofun(u) 
  r = pspre.alloc_r
  broadcast!(x->iofunction(x,pspre.neurontype),r,pspre.state_now)
  # multiply by weights, add to input of b .  input_b += W * r
  mul!(pspost.input,conn.weights,r,1,1)
  return nothing
end
