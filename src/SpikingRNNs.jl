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


"""
  next_poisson_spiketime(t_current::Float64,rate::Float64) -> t_next::Float64

  Returns next spike after current time `t_current` in a random Poisson process.
  with rate `rate`.
"""
@inline function next_poisson_spiketime(t_current::Float64,rate::Float64)
  return  t_current-log(rand())./rate
end


"""
  next_poisson_spiketime_from_function(t_current::Float64,fun_rate::Function,fun_rate_upper::Function; 
      Tmax::Float64=0.0,nowarning::Bool=false) -> Float64
  
  Returns the next spiketime in a Poisson process with time-varying rate. The rate variation is given by function `fun_rate`.
  
  See e.g.  Laub,Taimre,Pollet 2015
  
  # Arguments   
  + `t_start::Float64` : current time 
  + `fun_rate::Function` : `fun_rate(t::Float64) -> r::Float64` returns rate at time `t` 
  + `fun_rate_upper::Function` : upper limit to the function above. Strictly decreasing in `t`
     must be as close as possible to the `fun_rate` for efficiency
  + `Tmax::Float64` : upper threshold for spike proposal, maximum interval that can be produced    
  + `nowarning::Bool` : does not throw a warning when `Tmax`` is reached
"""
function next_poisson_spiketime_from_function(t_start::Float64,fun_rate::Function,fun_rate_upper::Function; 
    Tmax::Float64=50.0,nowarning::Bool=false)
  t = t_start 
  while (t-t_start) < Tmax 
    (rup::Float64) = fun_rate_upper(t)
    Δt = -log(rand())./rup # rand(Exponential())./rup
    t = t+Δt
    u = rand()*rup # rand(Uniform(0.0,rup))
    (_r::Float64) = fun_rate(t) 
    if u <= _r
      return t
    end
  end
  # if we are above Tmax, just return upper limit
  if !nowarning
    @warn "Upper limit reached, input firing below $(inv(Tmax)) Hz"
  end
  return Tmax + t_start
end


# legacy , to erase when possible!
_rand_by_thinning(t_start::Real,get_rate::Function,get_rate_upper::Function;
    Tmax=50.0,nowarning::Bool=false) = next_poisson_spiketime_from_function(t_start,get_rate,
      get_rate_upper;Tmax=Tmax,nowarning=nowarning)

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
#abstract type PopulationState{NT<:NeuronType} end
abstract type PopulationState end
nneurons(ps::PopulationState) = ps.n


# Everything related to connection (including plasticity, etc)
abstract type Connection end
abstract type PlasticityRule end

struct NoPlasticity <: PlasticityRule end
reset!(::NoPlasticity) = nothing

struct FakeConnection{N,PL<:NTuple{N,PlasticityRule}} <: Connection
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::PL
  function FakeConnection()
    w = sparse(fill(NaN,1,1))
    return new{0,NTuple{0,NoPlasticity}}(w,())
  end
end

abstract type AbstractBaseConnection <: Connection end
struct BaseConnection{N,PL<:NTuple{N,PlasticityRule}} <: Connection
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
struct ConnectionPlasticityTest{N,PL<:NTuple{N,PlasticityRule}} <: Connection
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


@inline function n_plasticity_rules(co::Connection)
  return length(co.plasticities)
end

struct PSSimpleInput{In} <: PopulationState
  neurontype::In # not really a neuron, but I keep the name for consistency
  n::Int64
  function PSSimpleInput(in::N) where N<:NeuronType
    return new{N}(in,1)
  end
end

function rand_pop_label()
  return Symbol(randstring(3))
end

abstract type AbstractPopulation{PS} end

# this is for populations without incoming synapses (so no presynaptic 
# population and no connections). Useful for input units.
struct UnconnectedPopulation{N,PS} <: AbstractPopulation{PS}
  label::Symbol
  state::PS
end
function UnconnectedPopulation(ps::PS;  
    label::Union{Nothing,String}=nothing) where PS<:PopulationState
  label = isnothing(label) ? rand_pop_label() : Symbol(label)
  return UnconnectedPopulation{0,PS}(label,ps)
end

@inline function n_prepops(::UnconnectedPopulation)
  return 0
end 


# struct Population{N,PS<:PopulationState,
#     TC<:NTuple{N,Connection},
#     TP<:NTuple{N,PopulationState}} <:AbstractPopulation{N,PS} 
#   label::Symbol # I use symbols because it might be a DataFrame column name
#   state::PS
#   connections::TC
#   pre_states::TP
# end

struct Population{PS<:PopulationState} <: AbstractPopulation{PS}
  label::Symbol # I use symbols because it might be a DataFrame column name
  state::PS
  connections::TC where {N,TC<:NTuple{N,Connection}}
  pre_states::TP  where {NN,TP<:NTuple{NN,PopulationState}}
end
nneurons(p::Population) = nneurons(p.state)

function Population(state::PopulationState,
    (conn_pre::Tuple{C,PS} where {C<:Connection,PS<:PopulationState})... ; 
    label::Union{Nothing,String,Symbol}=nothing)
  connections = Tuple(getindex.(conn_pre,1))
  pre_states = Tuple(getindex.(conn_pre,2))
  label = isnothing(label) ? rand_pop_label() : Symbol(label)
  return Population(label,state,connections,pre_states) 
end

@inline function n_prepops(p::Population)
  return length(p.pre_states)
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
function local_update!(::Real,::Real,::PopulationState)
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
  return nothing
end


##

include("rate_models.jl")
include("if_models.jl")
include("if_inputs.jl")
include("poisson_models.jl")

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