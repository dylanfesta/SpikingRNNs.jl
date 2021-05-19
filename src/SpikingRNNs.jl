module SpikingRNNs
using LinearAlgebra,Statistics,StatsBase,Random,Distributions
using SparseArrays
using ConfParser



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

function send_signal!(in::PopInputStatic)
  in.population_state.input .+= in.h
  return nothing
end

function reset_input!(ps::PopulationState)
  fill!(ps.input,0.0)
  return nothing
end

# this is the network iteration
function dynamics_step!(ntw::RecurrentNetwork)
  # update each population with the input already stored, reset inputs
  dynamics_step!.(ntw.dt,ntw.population_states)
  reset_input!.(ntw.population_states)
  # update the input of each postsynaptic population with the oputput of the presynaptic population
  send_signal!.(ntw.connections)
  # add the external inputs
  send_signal!.(ntw.inputs)
  # one iteration done!
  return nothing
end

include("./connectivity_utils.jl")
include("./rate_models.jl")

end # of SpikingRNNs module 