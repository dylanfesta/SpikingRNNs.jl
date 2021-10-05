# ./src/lif_conductance.jl

############

#=
coductance-based models, generalized so that they can have
different sypatic kernels (Exp or Expdiff)
and different spike generating functions (None, or Exp)
=#
abstract type SynapticKernel end
struct SKExp <: SynapticKernel
  τ::Float64
end
@inline function (sk::SKExp)(x::Real)
  return x/sk.τ
end
struct SKExpDiff <: SynapticKernel
  τ_plus::Float64
  τ_minus::Float64
end
@inline function (sk::SKExpDiff)(x_plus::Real,x_minus::Real)
  return (x_plus-x_minus) / (sk.τ_plus-sk.τ_minus)
end

abstract type SpikeGenFunction end
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


struct NTLIFConductance{SK<:SynapticKernel,SGen<:SpikeGenFunction} <: NeuronType
  synaptic_kernel::SK
  spikegen::SGen
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
    w::Real,pre_synker::SKExp,post_idx::Integer)
  # not normalized by taus  
	conn.post_trace1[post_idx] += w
  return nothing
end
@inline function trace_spike_update!(conn::ConnGeneralIF2,
    w::Float64,pre_synker::SKExpDiff,post_idx::Integer)
  # not normalized by taus  
	conn.post_trace1[post_idx] += w
	conn.post_trace2[post_idx] += w
  return nothing
end

# synaptic kernel as conductance
@inline function postsynaptic_kernel_update!(pspost::PopulationState,
   conn::ConnGeneralIF2,pspre::PSLIFConductance{NT},
   idx::Integer) where {SGen,NT<:NTLIFConductance{SKExp,SGen}}
	pspost.input[idx] += pspre.neurontype.synaptic_kernel(conn.post_trace1[idx]) * 
    (pspre.neurontype.v_reversal - pspost.state_now[idx])
  return nothing
end
@inline function postsynaptic_kernel_update!(pspost::PopulationState,
   conn::ConnGeneralIF2,pspre::PSLIFConductance{NT},
   idx::Integer) where {SGen,NT<:NTLIFConductance{SKExpDiff,SGen}}
  ker_term = pspre.neurontype.synaptic_kernel(conn.post_trace1[idx],conn.post_trace2[idx])
	pspost.input[idx] += ker_term*(pspre.neurontype.v_reversal - pspost.state_now[idx])
  return nothing
end


@inline function trace_decay!(conn::ConnGeneralIF2,dt::Real,
     pre_synkernel::SKExpDiff) 
	@inbounds @simd for i in eachindex(conn.post_trace1)
		conn.post_trace1[i] -= dt*conn.post_trace1[i] / pre_synkernel.τ_plus
		conn.post_trace2[i] -= dt*conn.post_trace2[i] / pre_synkernel.τ_minus
	end
  return nothing
end
@inline function trace_decay!(conn::ConnGeneralIF2,dt::Real,
     pre_synkernel::SKExp) 
	@inbounds @simd for i in eachindex(conn.post_trace1)
		conn.post_trace1[i] -= dt*conn.post_trace1[i] / pre_synkernel.τ
	end
  return nothing
end



## define two main functions here

function local_update!(t_now::Float64,dt::Float64,ps::PSLIFConductance)
	# computes the update to internal voltage, given the total input
  # dv =  (v_leak - v ) dt / τ + I dt / Cap
  dttau =  dt / ps.neurontype.τ_post
  dtCap = dt / ps.neurontype.capacitance
  # @show (ps.neurontype.v_leak[1] - ps.state_now[1])*dttau + ps.input[1]*dtCap
  # @show ps.neurontype.spikegen(ps.state_now[1])*dttau
  @. begin 
   ps.alloc_dv = (ps.neurontype.v_leak - ps.state_now)*dttau + ps.input*dtCap +
     ps.neurontype.spikegen(ps.state_now)*dttau
  end
  ps.state_now .+= ps.alloc_dv # v_{t+1} = v_t + dv
	# update spikes and refractoriness, and end
  return _spiking_state_update!(ps.state_now,ps.isfiring,ps.isrefractory,ps.last_fired,
    t_now,ps.neurontype.τ_refractory,ps.neurontype.v_threshold,ps.neurontype.v_reset)
end

function forward_signal!(t_now::Real,dt::Real,
      pspost::PSLIFConductance,conn::ConnGeneralIF2,pspre::PopulationState)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	#τ_decay = pspre.neurontype.τ_post_conductance_decay
  pre_synker = pspre.neurontype.synaptic_kernel
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
      trace_spike_update!(conn,weightsnz[_pnz],
        pre_synker,post_idx)
		end
	end
	# add the currents to postsynaptic input
	# ONLY non-refractory neurons
	post_refr = findall(pspost.isrefractory) # refractory ones
	@inbounds @simd for i in eachindex(pspost.input)
		if !(i in post_refr)
      postsynaptic_kernel_update!(pspost,conn,pspre,i)
      # input += conductance(t) * ( v_reversal - v(t) )
			#pspost.input[i] += 
      #  conn.post_trace[i] * (pspre.neurontype.v_reversal - pspost.state_now[i])

		end
	end
  # finally, all postsynaptic conductances decay in time
  trace_decay!(conn,dt,pspre.neurontype.synaptic_kernel)
	#@inbounds @simd for i in eachindex(conn.post_conductance)
	#	conn.post_trace[i] -= dt*conn.post_conductance[i] / τ_decay
	#end
  return nothing
end



#=
# threshold-linear input-output function
struct NTLIFCO <: NeuronType
  τ::Float64 # time constant
  Cap::Float64 # capacitance
  v_threshold::Vector{Float64} # spiking threshold (can vary among neurons)
  v_reset::Float64 # reset after spike
  v_leak::Float64 # reversal potential for leak term
  τ_refractory::Float64 # refractory time
  τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
  v_reversal::Float64 # reversal potential that affects postsynaptic neurons
end

struct PSLIFCO{NT} <: PopulationState{NT}
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
function PSLIFCO(p::NTLIFCO,n)
  zz() = zeros(Float64,n)
  ff() = falses(n)
  PSLIFCO(p,n,ntuple(zz,4)...,ntuple(ff,2)...,ntuple(zz,4))
end

# connection 
# must keep track of conductances 

struct ConnLIFCO{N,TP<:NTuple{N,PlasticityRule}} <: Connection
  weights::SparseMatrixCSC{Float64,Int64}
  post_conductance::Vector{Float64}
  plasticities::TP
end
function ConnLIFCO(weights::SparseMatrixCSC)
  npost=size(weights,2)
  ConnLIFCO(weights,zeros(Float64,npost),())
end

## define two main functions here

function local_update!(t_now::Float64,dt::Float64,ps::PSLIFCO)
	# computes the update to internal voltage, given the total input
  # dv =  (v_leak - v ) dt / τ + I dt / Cap
  dttau =  dt / ps.neurontype.τ
  dtCap = dt / ps.neurontype.Cap
  @. ps.alloc_dv = (ps.neurontype.v_leak - ps.state_now)*dttau + ps.input*dtCap
  ps.state_now .+= ps.alloc_dv # v_{t+1} = v_t + dv
	# update spikes and refractoriness, and end
  return _spiking_state_update!(ps.state_now,ps.isrefractory,ps.last_fired,
    t_now,ps.neurontype.τ_refractory,ps.neurontype.v_threshold,ps.neurontype.v_reset)
end


function forward_signal!(t_now::Real,dt::Real,
      pspost::PSLIFCO,conn::ConnectionLIFCO,pspre::PopulationState)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	τ_decay = pspre.neurontype.τ_post_conductance_decay
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
		  conn.post_conductance[post_idx] += weightsnz[_pnz] / τ_decay
		end
	end
	# add the currents to postsynaptic input
	# ONLY non-refractory neurons
	post_refr = findall(pspost.isrefractory) # refractory ones
	@inbounds @simd for i in eachindex(pspost.input)
		if !(i in post_refr)
      # input += conductance(t) * ( v_reversal - v(t) )
			pspost.input[i] += 
        conn.post_conductance[i] * (pspre.neurontype.v_reversal - pspost.state_now[i])
		end
	end
  # finally, all postsynaptic conductances decay in time
	@inbounds @simd for i in eachindex(conn.post_conductance)
		conn.post_conductance[i] -= dt*conn.post_conductance[i] / τ_decay
	end
  return nothing
end

=#