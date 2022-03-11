




struct ConnectionIFInput{S<:SynapticKernel}
  synaptic_kernel::S
  weights::Vector{Float64}
end



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
abstract type AbstractIFInputSpikes <: PopulationState


struct IFInputSpikesConstant{V<:Union{Float64,Vector{Float64}}} <: AbstractIFInputSpikes
  rate::V
  t_last_spike::Vector{Float64}
end
struct IFInputSpikesFunScalar <: AbstractIFInputSpikes
  f::Function # f(::Float64) -> Float64 
  t_last_spike::Vector{Float64}
end
struct IFInputSpikesFunVector <: AbstractIFInputSpikes
  f::Function # f(::Float64) -> Array{Float64}
  t_last_spike::Vector{Float64}
end



# Forward signals that arrive in the form of spikes 
function forward_signal!(::Real,dt::Real,
      pspost::PSIFNeuron,conn::ConnectionIFInput,pspre::AbstractIFInputSpikes)
	
  for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
      synaptic_kernel_trace_update!(conn,weightsnz[_pnz],post_idx)
		end
	end
	# add the currents to postsynaptic input vector ONLY non-refractory neurons
  # the term added depends on the neuron type, in general it is a function of the 
  # postsynaptic voltage
  add_signal_to_nonrefractory!(pspost.input,conn,pspost.isrefractory,pspost.state_now)

  # finally, all postsynaptic conductances decay in time
  kernel_decay!(dt,conn)
  return nothing
end


# HERE FOR REF!
function forward_signal!(t_now::Real,dt::Real,
      pspost::PSIFNeuron,conn::ConnectionIFInput,
      pspre::PSInputPoissonConductanceExact)
  preneu = pspre.neurontype
  pre_v_reversal = preneu.v_reversal
  pre_synker = preneu.synaptic_kernel
  sgen = pspre.neurontype.spikegenerator
  # traces time decay (here or at the end? meh)
  trace_decay!(dt,pspre)
  # if t_now moved past a spiketime...
  @inbounds for i in eachindex(pspre.firingtimes)
    tspike = pspre.firingtimes[i]
    _n_spikes = 0
    # count spike occurrences
    while tspike <= t_now
      _n_spikes += 1
      tspike = _get_spiketime_update(tspike,sgen,i)
    end
    # increment traces of spiking neurons
    # ASSUMING IT IS LINEAR
    if _n_spikes > 0
    # function defined in lif_conductance.jl
      trace_spike_update!(_n_spikes*pspre.input_weights[i],
          pspre.trace1,pspre.trace2,pre_synker,i)
      pspre.firingtimes[i]=tspike
    end
  end
  @inbounds @simd for i in eachindex(pspost.input)
    if ! pspost.isrefractory[i]
    # function defined in lif_conductance.jl
    postsynaptic_kernel_update!(pspost.input,pspost.state_now,
            pspre.trace1,pspre.trace2,pre_synker,pre_v_reversal,i)
    end
  end
  return nothing
end



#=

TO DO : spiking inputs 

TO DO : add global inhibition over here, as well
=#