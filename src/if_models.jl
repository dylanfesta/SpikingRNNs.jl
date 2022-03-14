

abstract type PSSpiking <: PopulationState end
abstract type SomaticKernel end
# TODO already defined elsewhere
#abstract type SynapticKernel end  
abstract type IFFiring end

abstract type AbstractConnectionIF{S} <: Connection end

struct SKLeak <: SomaticKernel
  v_leak::Float64
end
@inline function (sk::SKLeak)(v::Real)
  return (v-sk.v_leak)
end
struct SKLeakExp <: SomaticKernel
  v_leak::Float64
  v_th::Float64
  τ::Float64
end
@inline function (sk::SKLeakExp)(v::Real)
  return  (v-sk.v_leak) + sk.τ*exp((v-sk.vth)/sk.τ)
end

# Spiking mechanism : fixed threshold
struct IFFFixedThreshold
  v_threshold::Float64
  v_reset::Float64
  t_refractory::Float64
end

# I am skipping the neuron type struct, because it is really redundant
struct PSIFNeuron{So<:SomaticKernel,F<:IFFiring} <: PSSpiking
  τ::Float64
  capacitance::Float64
  somatic_kernel::So
  firing_process::F
  n::Int64
  input::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
end

function local_update!(t_now::Float64,dt::Float64,ps::PSIFNeuron)
	# computes the update to internal voltage, given the total input
  # dv =  somatic_function(v) dt / τ + I dt / Cap
  dttau =  dt / ps.neurontype.τ
  @inbounds for i in eachindex(ps.state_now)
    state_now = ps.state_now[i]
    state_now += (dttau*ps.somatic_kernel(state_now)+ps.input[i]*dtCap)
    ps.state_now[i] = state_now
  end
	# update spikes and refractoriness, and end
  process_spikes!(t_now,ps) # replaces _spiking_state_update!
  return nothing 
end


# Forward signals that arrive in the form of spikes 
function forward_signal!(::Real,dt::Real,
      pspost::PSIFNeuron,conn::AbstractConnectionIF,pspre::PSSpiking)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
      synaptic_kernel_trace_update!(conn.synaptic_kernel,weightsnz[_pnz],post_idx)
		end
	end
	# add the currents to postsynaptic input vector ONLY non-refractory neurons
  # the term added depends on the neuron type, in general it is a function of the 
  # postsynaptic voltage
  add_signal_to_nonrefractory!(pspost.input,conn,pspost.isrefractory,pspost.state_now)
  # finally, all postsynaptic conductances decay in time
  kernel_decay!(dt,conn.synaptic_kernel)
  return nothing
end

function process_spikes!(t_now::Real,ps::PSIFNeuron{SK,F}) where {SK,F<:IFFFixedThreshold}
  reset_spikes!(ps.isfiring)
  fp=ps.firing_process
	@inbounds @simd for i in eachindex(state_now)
    # remove refractoriness
    if ps.isrefractory[i] && ((t_now-last_fired[i]) >= fp.t_refractory)
			isrefractory[i] = false
		end
    # process spikes
		if ps.state_now[i] > fp.v_threshold
			ps.state_now[i] =  fp.v_reset
			ps.isfiring[i] = true
			ps.last_fired[i] = t_now
			ps.isrefractory[i] = true
    end
	end
  return nothing
end


struct ConnectionIF{S<:SynapticKernel} <: AbstractConnectionIF{S}
  synaptic_kernel::S
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::PL where {M,PL<:NTuple{M,PlasticityRule}}
end
function reset!(conn::ConnectionIF)
  reset!(conn.synaptic_kernel)
  reset!.(conn.plasticities)
end

struct SyKCurrentExponential <: SynapticKernel
  τ::Float64
  trace::Trace
  is_excitatory::Bool
end
reset!(sk::SyKCurrentExponential) = reset!(sk.trace)

struct SyKConductanceExponential <: SynapticKernel
  τ::Float64
  v_reversal::Float64
  trace::Trace
end
reset!(sk::SyKConductanceExponential) = reset!(sk.trace)

struct SyKConductanceDoubleExponential <: SynapticKernel
  τplus::Float64
  τminus::Float64
  v_reversal::Float64
  trace_plus::Trace
  trace_minus::Trace
end
reset!(sk::SyKConductanceDoubleExponential) = ( reset!(sk.trace_plus) ; reset!(sk.trace_minus))

# this is used for external input currents
struct SyKNothing <: SynapticKernel  end

@inline function kernel_decay!(dt::Real,
    sk::Union{SyKCurrentExponential,SyKConductanceExponential})
  trace_decay!(dt,sk.trace)  
end
@inline function kernel_decay!(dt::Real,sk::SyKConductanceDoubleExponential)
  trace_decay!(dt,sk.trace_plus)  
  trace_decay!(dt,sk.trace_minus)  
end

@inline function synaptic_kernel_trace_update!(syk::SyKCurrentExponential,w::Float64,idx::Integer)
  syk.trace[idx] += w
  return nothing
end
@inline function synaptic_kernel_trace_update!(syk::SyKConductanceExponential,w::Float64,idx::Integer)
  syk.trace[idx] += w
  return nothing
end
@inline function synaptic_kernel_trace_update!(syk::SyKConductanceDoubleExponential,w::Float64,idx::Integer)
  syk.trace_plus[idx] += w
  syk.trace_minus[idx] += w
  return nothing
end


@inline function add_signal_to_nonrefractory!(inputs::Vector{Float64},
      conn::AbstractConnectionIF{SyKCurrentExponential},isrefractory::BitArray{1},::Vector)
  syk = conn.synaptic_kernel
  if conn.is_excitatory
    @inbounds @simd for i in eachindex(inputs) 
    if ! isrefractory[i]
      inputs[i] .+=syk.trace[i]    
      end
    end
  else
    @inbounds @simd for i in eachindex(inputs) 
    if ! isrefractory[i]
      inputs[i] .-=syk.trace[i]    
      end
    end
  end
  return nothing
end
@inline function add_signal_to_nonrefractory!(inputs::Vector{Float64},
      conn::AbstractConnectionIF{SyKConductanceExponential},isrefractory::BitArray{1},
      state_now::Vector{Float64})
  syk = conn.synaptic_kernel
  @inbounds @simd for i in eachindex(inputs) 
  if ! isrefractory[i]
    inputs[i] .+= syk.trace[i]*(syk.v_reversal-state_now[i])
    end
  end
  return nothing
end
@inline function add_signal_to_nonrefractory!(inputs::Vector{Float64},
      syk::AbstractConnectionIF{SyKConductanceDoubleExponential},isrefractory::BitArray{1},
      state_now::Vector{Float64})
  error("NOT DONE YET!")
  return nothing
end
