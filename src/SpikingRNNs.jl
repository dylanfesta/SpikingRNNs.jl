module SpikingRNNs
using StatsBase: midpoint
using LinearAlgebra,Statistics,StatsBase,Random,Distributions
using SparseArrays
using ProgressLogging

# Base elements

abstract type Population end
abstract type PopulationState end
abstract type PopInput end

abstract type Connection end

struct RecurrentNetwork
  dt::Float64
  population_states::Tuple
  inputs::Tuple
  connections::Tuple
end

struct PopInputStatic{P<:PopulationState} <: PopInput
  population_state::P
  h::Vector{Float64}
end

@inline function send_signal!(t::Real,input::PopInputStatic)
  input.population_state.input .+= input.h
  return nothing
end

@inline function reset_input!(ps::PopulationState)
  fill!(ps.input,0.0)
  return nothing
end

@inline function reset_spikes!(spk::BitArray{1})
  fill!(spk,false)
	return nothing
end
@inline function reset_spikes!(popst::PopulationState)
	return reset_spikes!(popst.isfiring)
end


# this is the network iteration
function dynamics_step!(t_now::Float64,ntw::RecurrentNetwork)
  # update each population with the input already stored, reset inputs
  dynamics_step!.(t_now,ntw.dt,ntw.population_states)
  reset_input!.(ntw.population_states)
  # update the input of each postsynaptic population with the oputput of the presynaptic population
  send_signal!.(t_now,ntw.connections)
  # add the external inputs
  send_signal!.(t_now,ntw.inputs)
  # sometimes the connection variabiles evolve in time, too 
  dynamics_step!.(t_now,ntw.dt,ntw.connections)
  # one iteration done!
  return nothing
end

include("./connectivity_utils.jl")
include("./rate_models.jl")
include("./lif_current.jl")
include("./hawkes.jl")


end # of SpikingRNNs module 