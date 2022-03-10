

abstract type PSSpiking <: PopulationState end

abstract type SomaticKernel end
#abstract type SynapticKernel end
abstract type IFFiring end


# somatic kernels are leak, and leakexp

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

function forward_signal!(::Real,dt::Real,
      pspost::PSIFNeuron,conn::ConnectionIF,pspre::PSSpiking)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
      synaptic_kernel_trace_update!(conn,weightsnz[_pnz],post_idx)
		end
	end
	# add the currents to postsynaptic input vector
	# ONLY non-refractory neurons
	@inbounds @simd for i in eachindex(pspost.input)
		if !pspost.isrefractory[i]
      # replaces postsynaptic_kernel_update
      add_the_signal!(pspost.input,conn,i)
		end
	end
  # finally, all postsynaptic conductances decay in time
  kernel_decay!(dt,conn)
  return nothing
end


function process_spikes!(t_now::Real,
      ps::PSIFNeuron{SK,F}) where {SK,F<:IFFFixedThreshold}
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

#= 
 TO DOS

 new connection type should contain the synaptic kernel, 
 and the relative traces.

 synaptic kernel trace update as convolution of kernel and spike train
 exp or double exponential

 kernel_decay! just reduces the traces

 add_the_signal depends on whether connection is conductance based or current based

... and you are done!
=#