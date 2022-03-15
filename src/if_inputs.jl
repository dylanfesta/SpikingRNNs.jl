
struct ConnectionIFInput{S<:SynapticKernel} <: AbstractConnectionIF{S}
  synaptic_kernel::S
  weights::Vector{Float64}
end

ConnectionIFInput(weights::Vector{Float64}) = ConnectionIFInput(SyKNone(),weights)

# deals with inputs that are just currents
struct IFInputCurrentConstant{V<:Union{Float64,Vector{Float64}}} <: PopulationState
  current::V
end
struct IFInputCurrentFunScalar <: PopulationState
  f::Function # f(::Float64) -> Float64 
end
struct IFInputCurrentFunVector <: PopulationState
  f::Function # f(::Float64) -> Array{Float64}
end

function forward_signal!(::Real,dt::Real,
        pspost::PSIFNeuron,conn::ConnectionIFInput{SyKNothing},
        pspre::IFInputCurrentConstant{Float64})
  @inbounds @simd for i in eachindex(inputs) 
  if ! pspost.isrefractory[i]
    pspost.input[i] .+= conn.weights[i]*pspre.current
    end
  end
  return nothing
end
function forward_signal!(::Real,::Real,
        pspost::PSIFNeuron,conn::ConnectionIFInput{SyKNothing},
        pspre::IFInputCurrentConstant{<:Vector})
  @inbounds @simd for i in eachindex(inputs) 
  if ! pspost.isrefractory[i]
    pspost.input[i] .+= conn.weights[i]*pspre.current[i]
    end
  end
  return nothing
end
function forward_signal!(t_now::Real,::Real,
        pspost::PSIFNeuron,conn::ConnectionIFInput{SyKNothing},
        pspre::IFInputCurrentFunScalar)
  curr_now::Float64 = pspre.f(t_now)
  @inbounds @simd for i in eachindex(inputs) 
  if ! pspost.isrefractory[i]
    pspost.input[i] .+= conn.weights[i]*curr_now
    end
  end
  return nothing
end
function forward_signal!(t_now::Real,::Real,
        pspost::PSIFNeuron,conn::ConnectionIFInput{SyKNothing},
        pspre::IFInputCurrentFunVector)
  curr_now::Vector{Float64} = pspre.f(t_now)
  @inbounds @simd for i in eachindex(inputs) 
  if ! pspost.isrefractory[i]
    pspost.input[i] .+= conn.weights[i]*curr_now[i]
    end
  end
  return nothing
end

# Now exact spiking inputs
abstract type AbstractIFInputSpikes <: PopulationState end

struct IFInputSpikesConstant{V<:Union{Float64,Vector{Float64}}} <: AbstractIFInputSpikes
  rate::V
  t_last_spike::Vector{Float64}
end
struct IFInputSpikesFunScalar <: AbstractIFInputSpikes
  f::Function # f(::Float64) -> Float64 
  f_upper::Function
  t_last_spike::Vector{Float64}
end
struct IFInputSpikesFunVector <: AbstractIFInputSpikes
  f::Function # f(::Float64,idx::Integer) -> Float64
  f_upper::Function
  t_last_spike::Vector{Float64}
end
struct IFInputSpikesTrain <: AbstractIFInputSpikes
  train::Vector{Vector{Float64}}
  counter::Vector{Int64}
  t_last_spike::Vector{Float64}
end

function reset!(in::AbstractIFInputSpikes)
  fill!(in.t_last_spike,-Inf)
  return nothing
end
function reset!(in::IFInputSpikesTrain)
  fill!(in.t_last_spike,-Inf)
  fill!(in.counter,0)
  return nothing
end

# Forward signals that arrive in the form of spikes 
function forward_signal!(t_now::Real,dt::Real,
      pspost::PSIFNeuron,conn::ConnectionIFInput,pspre::AbstractIFInputSpikes)
  for i in eachindex(pspre.t_last_spike)
    # counts number of spikes between stored last spike, and t_now
    tspike = pspre.t_last_spike[i]
    _n_spikes = 0
    while tspike <= t_now
      _n_spikes += 1
      tspike = get_next_input_spiketime(tspike,pspre,i)
    end
    # applies spikes
    if _n_spikes > 0
      synaptic_kernel_trace_update!(conn.synaptic_kernel,_n_spikes*conn.weights[i],i)
      pspre.t_last_spike[i] = tspike
    end
  end
  add_signal_to_nonrefractory!(pspost.input,conn,pspost.isrefractory,pspost.state_now)
  kernel_decay!(dt,conn.synaptic_kernel)
  return nothing
end


@inline function get_next_input_spiketime(tnow::Real,ps::IFInputCurrentConstant{Float64},::Integer)
  return tnow - log(rand())/ps.rate  # rand(Exponential())
end
@inline function get_next_input_spiketime(tnow::Real,ps::IFInputCurrentConstant{Vector{Float64}},i::Integer)
  return tnow - log(rand())/ps.rate[i]
end
@inline function get_next_input_spiketime(tnow::Real,ps::IFInputSpikesFunScalar,::Integer)
  return next_poisson_spiketime_from_function(tnow,ps.f,ps.f_upper)
end
@inline function get_next_input_spiketime(tnow::Real,ps::IFInputSpikesFunVector,i::Integer)
  f(t) = ps.f(t,i)
  f_upper(t) = ps.f_upper(t,i)
  return next_poisson_spiketime_from_function(tnow,f,f_upper)
end

function get_next_input_spiketime(::Real,ps::IFInputSpikesTrain,i::Integer)
  # note that t_current_spike is expected to be 
  # sg.trains[i][counter[i]] (before counter update)
  c = ps.counter[i]
  if checkbounds(Bool,ps.trains[i],c+1) 
    ps.counter[i] = c+1    # move counter forward
    return ps.trains[i][c+1] # return next spike time
  else
    return Inf
  end 
end

# TO DO : global inhibition