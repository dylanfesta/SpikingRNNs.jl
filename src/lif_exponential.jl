# ./src/lif_conductance.jl

############
# conductance-based (classic) LIF

# threshold-linear input-output function
struct NTEIF <: NeuronType
  τ::Float64 # time constant (membrane capacitance)
  g_l::Float64 # conductance leak
  v_expt::Float64 # exp term threshold
  steep_exp::Float64 # steepness of exponential term
  v_threshold::Float64 # spiking threshold 
  v_reset::Float64 # reset after spike
  v_leak::Float64 # reversal potential for leak term
  τ_refractory::Float64 # refractory time
end

struct PSEIF{NT} <: PopulationState{NT}
  neurontype::NT
  n::Int64 # pop size
  state_now::Vector{Float64}
  input::Vector{Float64}
	alloc_dv::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
  function PSEIF(p::NTEIF,n::Integer)
    zz = _ -> zeros(Float64,n)
    ff = _ -> falses(n)
    new{NTEIF}(p,n,ntuple(zz,4)...,ntuple(ff,2)...)
  end
end

# connection, can use ConnectionLIF


## define two main functions here

function local_update!(t_now::Float64,dt::Float64,ps::PSEIF)
	# computes the update to internal voltage, given the total input
  # dv = (dt / τ) * (g_l (v_leak - v ) + g_l * steep_exp * exp((v-v_expt)/steep_exp)
  #  + input ) 
  dttau =  dt / ps.neurontype.τ
  @. ps.alloc_dv =  dttau * ( ps.neurontype.g_l*(ps.neurontype.v_leak-ps.state_now)
    + ps.neurontype.steep_exp*ps.neurontype.g_l*exp(ps.state_now-ps.neurontype.v_expt)
    + ps.input)
  ps.state_now .+= ps.alloc_dv # v_{t+1} = v_t + dv
	# update spikes and refractoriness, and end
  return _spiking_state_update!(ps.state_now,ps.isfiring,ps.isrefractory,ps.last_fired,
    t_now,ps.neurontype.τ_refractory,ps.neurontype.v_threshold,ps.neurontype.v_reset)
end


function forward_signal!(t_now::Real,dt::Real,
      pspost::PSEIF,conn::ConnLIF,pspre::PopulationState)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	τ_decay = pspre.neurontype.τ_post_current_decay
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
			# update the post currents even when refractory
			conn.post_current[post_idx] += 
						weightsnz[_pnz] / τ_decay
		end
	end
	# add the currents to postsynaptic input
	# ONLY non-refractory neurons
	post_refr = findall(pspost.isrefractory) # refractory ones
	@inbounds @simd for i in eachindex(pspost.input)
		if !(i in post_refr)
			pspost.input[i] += conn.post_current[i]
		end
	end
  # finally, postsynaptic currents decay in time
	@inbounds @simd for i in eachindex(conn.post_current)
		conn.post_current[i] -= dt*conn.post_current[i] / τ_decay
	end
  return nothing
end

