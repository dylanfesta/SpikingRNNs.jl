module SpikingRNNs
using LinearAlgebra,Statistics,StatsBase,Random,Distributions
using SparseArrays
using ProgressLogging

# Base elements

# neuron parameterns and behavior
abstract type NeuronType end
Base.broadcastable(t::NeuronType)=Ref(t)


# Dynamical state and memory allocation for a particular group of neurons
# n is always the number of neurons, and neurontype is the neuron type
abstract type PopulationState{NT<:NeuronType} end
nneurons(ps::PopulationState) = ps.n


# Everything related to connection (including plasticity, etc)
abstract type Connection end

abstract type AbstractPopulation end

struct Population{N,PS<:PopulationState,
    TC<:NTuple{N,Connection},TP<:NTuple{N,PopulationState} } <:AbstractPopulation 
  label::Symbol # I use symbols because it might be a DataFrame column name
  state::PS
  connections::TC
  pre_states::TP
end
nneurons(p::Population) = nneurons(p.state)
function Population(label::String,state::PopulationState,connections,pre_states)
  return Population(Symbol(label),state,connections,pre_states) 
end
function Population(state::PopulationState,connections,pre_states)
  return Population(Symbol(randstring(3)),state,connections,pre_states) 
end


@inline function n_prepops(p::Population{N,PS,TC,TP}) where {N,PS,TC,TP}
  return N
end 

# this is for populations without incoming synapses (so no presynaptic 
# population and no connections). Useful for input units.
struct UnconnectedPopulation{PS<:PopulationState} <:AbstractPopulation
  state::PS
end

abstract type AbstractNetwork end

struct RecurrentNetwork{N,TP<:NTuple{N,AbstractPopulation}} <: AbstractNetwork
  dt::Float64
  populations::TP
end

# connection and plasticity 

abstract type PlasticityRule end
struct NoPlasticity <: PlasticityRule end

struct BaseConnection{N,TP<:NTuple{N,PlasticityRule}} <: Connection
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::TP
end

function BaseConnection(w::SparseMatrixCSC)
  return BaseConnection{0,NTuple{0,NoPlasticity}}(w,())
end


# fallback functions

# updates the state of the population locally, depending on
# some input gathered in previous steps
# usually it's simply the vector p.input
function local_update!(tnow::Real,dt::Real,p::PopulationState)
  return nothing  
end
function local_update!(tnow::Real,dt::Real,p::Population)
  return local_update!(tnow,dt,p.state)
end

# reset the input gathered in the previous steps
# usually it's simply the vector p.input
@inline function reset_input!(ps::PopulationState)
  fill!(ps.input,0.0)
  return nothing
end
function reset_input!(p::Population)
  return reset_input!(p.state)
end
# or sometimes there are spikes
# stored in the BitVector called `isfiring`
@inline function reset_spikes!(spk::BitArray{1})
  fill!(spk,false)
	return nothing
end
@inline function reset_spikes!(popst::PopulationState)
	return reset_spikes!(popst.isfiring)
end

# Computes the input, considering each connection and each presynaptic population
function forward_signal!(tnow::Real,dt::Real,p_post::PopulationState,c::Connection,p_pre::PopulationState)
  return nothing
end
function forward_signal!(tnow::Real,dt::Real,p::Population)
  for i in 1:n_prepops(p)
    forward_signal!(tnow,dt,p.state,p.connections[i],p.pre_states[i])
  end
  return nothing  
end



# this is the network iteration
function dynamics_step!(t_now::Float64,ntw::RecurrentNetwork)
  reset_input!.(ntw.populations)
  forward_signal!.(t_now,ntw.dt,ntw.populations)
  local_update!.(t_now,ntw.dt,ntw.populations)
  return nothing
end

dynamics_step!(ntw::RecurrentNetwork) = dynamics_step!(NaN,ntw)

include("connectivity_shared.jl")
include("inputs_shared.jl")
include("rate_models.jl")
include("firingneurons_shared.jl")
include("lif_current.jl")
include("lif_exponential.jl")
include("recorders_shared.jl")

#=
include("./hawkes.jl")
=#

end # of SpikingRNNs module 