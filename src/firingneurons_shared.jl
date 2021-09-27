
# this is for presynaptic , covers also Poisson inputs generators, etc
abstract type PSSpikingType{NT} <: PopulationState{NT} end

# this is for postsynaptic that integrate inputs over currents
# LIF, EIF, QIF, etc
abstract type PSGeneralCurrentIFType{NT} <: PSSpikingType{NT} end


# connection for IF neurons 
# must keep track of internal trace (current/conductance)

struct ConnGeneralIF{N,TP<:NTuple{N,PlasticityRule}} <: Connection{N}
  weights::SparseMatrixCSC{Float64,Int64}
  post_trace::Vector{Float64}
  plasticities::TP
end

function ConnGeneralIF(weights::SparseMatrixCSC)
  npost=size(weights,2)
  return ConnGeneralIF(weights,zeros(Float64,npost),())
end
function ConnGeneralIF(weights::SparseMatrixCSC,
    (plasticities::PlasticityRule)...)
  npost=size(weights,2)
  return ConnGeneralIF(weights,zeros(Float64,npost),plasticities)
end

# This is an "infinitely strong" voltage connection
# if pre spikes, and *any* weight is present
# then  post will also spike (no delay for now)
# it cannot be plastic, but one can still modify the connectivity matrix
struct ConnSpikeTransfer{N,TP<:NTuple{N,PlasticityRule}} <: Connection{N}
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::TP
  function ConnSpikeTransfer(weights::SparseMatrixCSC)
    return  new{0,Tuple{}}(weights,())
  end
end

# see ConnSpikeTransfer comments
function forward_signal!(t_now::Real,dt::Real,
      pspost::PSGeneralCurrentIFType,
      conn::ConnSpikeTransfer,pspre::PSSpikingType)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
      post_idx = post_idxs[_pnz]
			pspost.input[post_idx] = Inf
		end
	end
  return nothing
end


function forward_signal!(t_now::Real,dt::Real,
      pspost::PSGeneralCurrentIFType,
      conn::ConnGeneralIF,pspre::PSSpikingType)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	τ_decay = pspre.neurontype.τ_output_decay
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
			# update the post currents even when refractory
			conn.post_trace[post_idx] += 
						weightsnz[_pnz] / τ_decay
		end
	end
	# add the currents to postsynaptic input
	# ONLY non-refractory neurons
	post_refr = findall(pspost.isrefractory) # refractory ones
	@inbounds @simd for i in eachindex(pspost.input)
		if !(i in post_refr)
			pspost.input[i] += conn.post_trace[i]
		end
	end
  # finally, postsynaptic currents decay in time
	@inbounds @simd for i in eachindex(conn.post_trace)
		conn.post_trace[i] -= dt*conn.post_trace[i] / τ_decay
	end
  return nothing
end


# examines state_now, if above threshold, isfiring and refractory turn true
# and potential is reset. Finally, removes expired refractoriness
function _spiking_state_update!(state_now::Vector{R},
    isfiring::BitArray{1},isrefractory::BitArray{1},
    last_fired::Vector{R},
    t_now::R, t_refractory::R,
    v_threshold::R, v_reset::R) where R<:Real
  reset_spikes!(isfiring)  
	@inbounds @simd for i in eachindex(state_now)
		if state_now[i] > v_threshold
			state_now[i] =  v_reset
			isfiring[i] = true
			last_fired[i] = t_now
			isrefractory[i] = true
		# check only when refractory
		elseif isrefractory[i] && 
				( (t_now-last_fired[i]) >= t_refractory)
			isrefractory[i] = false
		end
	end
  return nothing
end