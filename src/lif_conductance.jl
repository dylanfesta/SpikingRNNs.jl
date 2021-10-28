# ./src/lif_conductance.jl

############

#=
coductance-based models, generalized so that they can have
different sypatic kernels (Exp or Expdiff)
and different spike generating functions (None, or Exp)
=#
# abstract type SynapticKernel end moved to main
struct SKExp <: SynapticKernel
  τ::Float64
end

# Input update for conductance based IF neurons, assuming the kernel takes
# at most two traces arguments (as in diff of exponentials)  
@inline function postsynaptic_kernel_update!(input::Vector{R},post_state::Vector{R},
    post_trace1::Vector{R},post_trace2::Vector{R},pre_synkernel::SynapticKernel,
    pre_v_reversal::R,idx::Integer) where R
  ker_term = pre_synkernel(post_trace1[idx],post_trace2[idx])
	input[idx] += ker_term*(pre_v_reversal - post_state[idx])
  return nothing
end

# # this function is not needed
# @inline function (sk::SKExp)(x::Real)
#   return x/sk.τ
# end
@inline function (sk::SKExp)(x::Real,::Real)
  return x/sk.τ
end
@inline function trace_spike_update!(w::R,trace1::Vector{R},::Vector{R},
    ::SKExp,idx::Integer) where R
  trace1[idx] += w  
end

# difference of exponentials, needs two traces
struct SKExpDiff <: SynapticKernel
  τ_plus::Float64
  τ_minus::Float64
end
@inline function (sk::SKExpDiff)(x_plus::Real,x_minus::Real)
  return (x_plus-x_minus) / (sk.τ_plus-sk.τ_minus)
end
@inline function trace_spike_update!(w::R,trace1::Vector{R},trace2::Vector{R},
    ::SKExpDiff,idx::Integer) where R
  trace1[idx] += w  
  trace2[idx] += w  
end


# abstract type SpikeGenFunction end # moved to main file
struct SpikeGenNone <: SpikeGenFunction end
@inline function (sk::SpikeGenNone)(::R) where R
  return zero(R)
end

struct SpikeGenEIF <: SpikeGenFunction
  vth_exp::Float64
  s::Float64
end
@inline function (sk::SpikeGenEIF)(v::R) where R<:Real
  return sk.s*exp((v-sk.vth_exp)/sk.s)
end

# abstract type NTConductance <:NeuronType end moved to main file
# ALL conductance neurons must have spikes (neuron.isfiring::BitArray{1}) 
# a synaptic kernel (neuron.synaptic_kernel::SynapticKernel) 
# and a reversal potential (neuron.v_reversal::Float64)

struct NTLIFConductance{SK<:SynapticKernel} <: NTConductance
  synaptic_kernel::SK
  spikegen::SpikeGenFunction
  τ_post::Float64 # time constant for postsynaptic update
  capacitance::Float64 # membrane capacitance
  v_threshold::Float64 # max voltage before reset
  v_reset::Float64 # reset after spike
  v_leak::Float64 # reversal potential for leak term
  τ_refractory::Float64 # refractory time
  v_reversal::Float64 # reversal potential that affects postsynaptic neurons
end

struct PSLIFConductance{NT} <: PSSpikingType{NT}
  neurontype::NT
  n::Int64 # pop size
  state_now::Vector{Float64}
  input::Vector{Float64}
	alloc_dv::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
	pre_reverse_potentials::Vector{Float64}
	pre_conductances_now::Vector{Float64}
end
function PSLIFConductance(p::NTLIFConductance,n)
  zz = _ -> zeros(Float64,n)
  ff = _ ->  falses(n)
  PSLIFConductance(p,n,ntuple(zz,4)...,ntuple(ff,2)...,ntuple(zz,2)...)
end

function reset!(ps::PSLIFConductance)
  fill!(ps.state_now,0.0)
  fill!(ps.input,0.0)
  fill!(ps.last_fired,-1E6)
  fill!(ps.isfiring,false)
  fill!(ps.isrefractory,false)
  fill!(ps.pre_reverse_potentials,0.0)
  fill!(ps.pre_conductances_now,0.0)
  return nothing
end

# connection 
# use ConnGeneralIF2

@inline function trace_spike_update!(conn::ConnGeneralIF2,
    w::Real,pre_sker::SynapticKernel,post_idx::Integer)
  # not normalized by taus  
  return trace_spike_update!(w,conn.post_trace1,conn.post_trace2,pre_sker,post_idx)
end

@inline function postsynaptic_kernel_update!(pspost::PopulationState,
     conn::ConnGeneralIF2,pre_synkernel::SynapticKernel,pre_v_reversal::Float64,idx::Integer)
  return postsynaptic_kernel_update!(pspost.input,pspost.state_now,
    conn.post_trace1,conn.post_trace2,pre_synkernel,
    pre_v_reversal,idx)
end

@inline function trace_decay!(dt::Real,conn::ConnGeneralIF2,synker::SynapticKernel)
  return trace_decay!(dt,conn.post_trace1,conn.post_trace2,synker)
end
@inline function trace_decay!(dt::R,tr1::Vector{R},::Vector{R},sk::SKExp) where R
  tr1 .*= exp(-dt/sk.τ)
  return nothing
end
@inline function trace_decay!(dt::R,tr1::Vector{R},tr2::Vector{R},sk::SKExpDiff) where R
  tr1 .*= exp(-dt/sk.τ_plus)
  tr2 .*= exp(-dt/sk.τ_minus)
  return nothing
end


## define two main functions here
function local_update!(t_now::Float64,dt::Float64,ps::PSLIFConductance)
	# computes the update to internal voltage, given the total input
  # dv =  (v_leak - v ) dt / τ + I dt / Cap
  dttau =  dt / ps.neurontype.τ_post
  dtCap = dt / ps.neurontype.capacitance
  @inbounds for i in eachindex(ps.state_now)
    state_now = ps.state_now[i]
    state_now += (  (ps.neurontype.v_leak - state_now)*dttau + 
      ps.input[i]*dtCap +
      ps.neurontype.spikegen(state_now)*dttau )
    ps.state_now[i] = state_now
  end
	# update spikes and refractoriness, and end
  return _spiking_state_update!(ps.state_now,ps.isfiring,ps.isrefractory,ps.last_fired,
    t_now,ps.neurontype.τ_refractory,ps.neurontype.v_threshold,ps.neurontype.v_reset)
end

function forward_signal!(t_now::Real,dt::Real,
      pspost::PSLIFConductance,conn::ConnGeneralIF2,pspre::PopulationState)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
  pre_synker = pspre.neurontype.synaptic_kernel
  pre_v_reversal = pspre.neurontype.v_reversal
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
      trace_spike_update!(conn,weightsnz[_pnz],pre_synker,post_idx)
		end
	end
	# add the currents to postsynaptic input
	# ONLY non-refractory neurons
	@inbounds @simd for i in eachindex(pspost.input)
		if ! pspost.isrefractory[i]
      postsynaptic_kernel_update!(pspost,conn,pre_synker,pre_v_reversal,i)
		end
	end
  # finally, all postsynaptic conductances decay in time
  trace_decay!(dt,conn,pre_synker)
  return nothing
end
