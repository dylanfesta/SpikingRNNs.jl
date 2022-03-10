module SpikingRNNs
using LinearAlgebra,Statistics,StatsBase,Random,Distributions
using SparseArrays
using Colors # to save spike rasters as png
using DSP # this is to have convolutions to analyze spike trains
using ExtendableSparse # to optimize adding elements in sparse matrices

# small general utility functions

"""
  hardbounds(x::R,low::R,high::R) where R = min(high,max(x,low))

Applies hard-bounds on scalar `x`  
"""
@inline hardbounds(x::R,low::R,high::R) where R = min(high,max(x,low))


# all type declarations should go here :-/

abstract type NeuronType end
abstract type SynapticKernel end
abstract type NTConductance <:NeuronType end
abstract type SpikeGenFunction end 

# Base elements

# neuron parameterns and behavior
Base.broadcastable(t::NeuronType)=Ref(t)


# Dynamical state and memory allocation for a particular group of neurons
# n is always the number of neurons, and neurontype is the neuron type
abstract type PopulationState{NT<:NeuronType} end
nneurons(ps::PopulationState) = ps.n


# Everything related to connection (including plasticity, etc)
abstract type Connection{N} end
abstract type PlasticityRule end

struct NoPlasticity <: PlasticityRule end
reset!(::NoPlasticity) = nothing

struct FakeConnection{N,PL<:NTuple{N,PlasticityRule}} <: Connection{N}
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::PL
  function FakeConnection()
    w = sparse(fill(NaN,1,1))
    return new{0,NTuple{0,NoPlasticity}}(w,())
  end
end

struct BaseConnection{N,PL<:NTuple{N,PlasticityRule}} <: Connection{N}
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::PL
end
function BaseConnection(w::SparseMatrixCSC)
  return BaseConnection{0,NTuple{0,NoPlasticity}}(w,())
end

# weights evolve according to plasticity rules and pre-post activity
# BUT neurons do not exchange signals of any kind
# this works well in tandem with input neurons with specific spike times
# to test plasticity rules
struct ConnectionPlasticityTest{N,PL<:NTuple{N,PlasticityRule}} <: Connection{N}
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::PL
end
function ConnectionPlasticityTest(weights::SparseMatrixCSC,
    (plasticities::PlasticityRule)...)
  return ConnectionPlasticityTest(weights,plasticities)
end
function reset!(conn::ConnectionPlasticityTest)
  reset!.(conn.plasticities)
  return nothing
end


@inline function n_plasticity_rules(::Connection{N}) where N
  return N
end

struct PSSimpleInput{In} <: PopulationState{In}
  neurontype::In # not really a neuron, but I keep the name for consistency
  n::Int64
  function PSSimpleInput(in::N) where N<:NeuronType
    return new{N}(in,1)
  end
end

function rand_pop_label()
  return Symbol(randstring(3))
end

abstract type AbstractPopulation{N,PS} end

# this is for populations without incoming synapses (so no presynaptic 
# population and no connections). Useful for input units.
struct UnconnectedPopulation{N,PS} <: AbstractPopulation{N,PS}
  label::Symbol
  state::PS
end
function UnconnectedPopulation(ps::PS;  
    label::Union{Nothing,String}=nothing) where PS<:PopulationState
  label = isnothing(label) ? rand_pop_label() : Symbol(label)
  return UnconnectedPopulation{0,PS}(label,ps)
end


struct Population{N,PS<:PopulationState,
    TC<:NTuple{N,Connection},
    TP<:NTuple{N,PopulationState}} <:AbstractPopulation{N,PS} 
  label::Symbol # I use symbols because it might be a DataFrame column name
  state::PS
  connections::TC
  pre_states::TP
end
nneurons(p::Population) = nneurons(p.state)

function Population(state::PopulationState,
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:PopulationState})... ; 
    label::Union{Nothing,String}=nothing)
  connections = Tuple(getindex.(conn_pre,1))
  pre_states = Tuple(getindex.(conn_pre,2))
  label = isnothing(label) ? rand_pop_label() : Symbol(label)
  return Population(label,state,connections,pre_states) 
end

@inline function n_prepops(p::Population{N,PS,TC,TP}) where {N,PS,TC,TP}
  return N
end 
@inline function n_prepops(p::AbstractPopulation{N,PS}) where {N,PS}
  return N
end 

abstract type AbstractNetwork end
struct RecurrentNetwork{N,TP<:NTuple{N,AbstractPopulation}} <: AbstractNetwork
  dt::Float64
  populations::TP
end

function RecurrentNetwork(dt,(pops::P where P<:AbstractPopulation)...)
  RecurrentNetwork(dt,pops)
end

# fallback functions

# updates the state of the population locally, depending on
# some input gathered in previous steps
# usually it's simply the vector p.input
function local_update!(tnow::Real,dt::Real,p::PopulationState)
  return nothing  
end
function local_update!(tnow::Real,dt::Real,p::AbstractPopulation)
  return local_update!(tnow,dt,p.state)
end

# reset the input gathered in the previous steps
# usually it's simply the vector p.input
@inline function reset_input!(ps::PopulationState)
  fill!(ps.input,0.0)
  return nothing
end
function reset_input!(p::AbstractPopulation)
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
function forward_signal!(tnow::Real,dt::Real,p::AbstractPopulation)
  for i in 1:n_prepops(p)
    forward_signal!(tnow,dt,p.state,p.connections[i],p.pre_states[i])
  end
  return nothing  
end
function plasticity_update!(tnow::Real,dt::Real,p::AbstractPopulation)
  for i in 1:n_prepops(p)
    for r in 1:n_plasticity_rules(p.connections[i])
      plasticity_update!(tnow,dt,p.state,p.connections[i],
        p.pre_states[i],p.connections[i].plasticities[r])
    end
  end
  return nothing  
end


# this is the network iteration
function dynamics_step!(t_now::Float64,ntw::RecurrentNetwork)
  reset_input!.(ntw.populations)
  forward_signal!.(t_now,ntw.dt,ntw.populations)
  local_update!.(t_now,ntw.dt,ntw.populations)
  plasticity_update!.(t_now,ntw.dt,ntw.populations)
  return nothing
end

dynamics_step!(ntw::RecurrentNetwork) = dynamics_step!(NaN,ntw)

## Activity traces as fundamental building blocks

struct Trace
  val::Vector{Float64}
  τ::Float64
  function Trace(τ::Float64,n::Integer)
    val = fill(0.0,n)
    return new(val,τ)
  end
end
function nneurons(tra::Trace)
  return length(tra.val)
end

Base.getindex(tra::Trace,idx::Integer) = tra.val[idx]
Base.setindex!(tra::Trace,x::Float64,idx::Integer) = (tra.val[idx] = x)


function trace_decay!(tra::Trace,Δt::Float64)
  tra.val .*= exp(-Δt/tra.τ)
  return nothing
end

# slower than trace_decay! , please use trace_decay! instead
function trace_step!(tra::Trace,dt::Float64)
  val = tra.val
	@inbounds @simd for i in eachindex(val)
		val[i] -= dt*val[i] / tra.τ
	end
end

function reset!(tra::Trace)
  fill!(tra.val,0.0) 
  tra.t_last[]=0.0
  return nothing
end


##

include("rate_models.jl")
include("if_models.jl")

include("firingneurons_shared.jl")
include("firingneurons_plasticity_shared.jl")
include("firingneurons_heterosynaptic_plasticity.jl")
include("firingneurons_structural_plasticity.jl")
include("connectivity_shared.jl")
include("lif_current.jl")
include("lif_exponential.jl")
include("lif_conductance.jl")
include("inputs_shared.jl")
include("recorders_shared.jl")

end # of SpikingRNNs module 