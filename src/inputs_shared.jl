
##############
# Inputs 


# if the state has an input, this will sum
# a linear component to it at each step (a constant input current) 
# the scaling is regulated here, and the weight is ignored
struct InputSimpleOffset <: NeuronType
  α::Float64 # scaling constant
end
# independent Gaussian noise for each neuron
# as before, the scaling is in the weight
struct InputIndependentNormal <: NeuronType
  α::Float64 # scaling constant
end

struct PSSimpleInput{In} <: PopulationState{In}
  neurontype::In # not really a neuron, but I keep the name for consistency
  n::Int64
  function PSSimpleInput(in::N) where N<:NeuronType
    return new{N}(in,1)
  end
end


# when the presynaptic is a simple input, just sum linearly to the input vector
function forward_signal!(tnow::Real,dt::Real,p_post::PopulationState,
    c::Connection,p_pre::PSSimpleInput{InputSimpleOffset})
  for i in eachindex(p_post.input)
    p_post.input[i] += p_pre.neurontype.α
  end
  return nothing
end

# the connection is ignored: all neurons of the same population receive
# the same noise. Postsynaptic neurons must have a τ
function forward_signal!(tnow::Real,dt::Real,p_post::PopulationState,
    c::Connection,p_pre::PSSimpleInput{InputIndependentNormal})
  # std is α for isolated neuron
  _reg = sqrt(2*p_post.neurontype.τ/dt)
  for i in eachindex(p_post.input)
    p_post.input[i] += p_pre.neurontype.α*_reg*randn()
  end
  return nothing
end


# Poisson firing
struct NTPoisson <: NeuronType
  rate::Ref{Float64} # rate is in Hz and is mutable
  τ_post_current_decay::Float64 # decay of postsynaptic conductance
end

# conductance based Poisson firing
struct NTPoissonCO <: NeuronType
  rate::Ref{Float64} # rate is in Hz and is mutable
  τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
  v_reversal::Float64 # reversal potential that affects postsynaptic neurons
end

struct PSPoisson{NT} <: PopulationState{NT}
  neurontype::NT
  n::Int64 # pop size
	isfiring::BitArray{1} # firing will be i.i.d. Poisson
  isfiring_alloc::Vector{Float64} # allocate probabilities
  function PSPoisson(p::NT,n) where NT<:NeuronType
    new{NT}(p,n,falses(n),zeros(Float64,n))
  end
end

# >>> TO DO
# Note: to have local updates, you should use a ConnLIFCO_fixed from the Poisson
# to the connected population(s), but also include 
# an UnconnectedPopulation with the 
# Poissoin population state in it when you define the network.

# function local_update!(t_now::Float64,dt::Float64,ps::PSPoisson)
#   c = dt*ps.rate[]
#   @assert c < 1 "Frequency of Poisson input too high!"
#   rand!(ps.isfiring_alloc)
#   map!(r->r < c, ps.isfiring, ps.isfiring_alloc)
#   return nothing
# end

# # forward_signal!(...)  no need to redefine it! 