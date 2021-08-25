# ./src/lif_conductance.jl

############
# conductance-based (classic) LIF

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
function PSLIFCO(p::Population,n)
  zz() = zeros(Float64,n)
  ff() = falses(n)
  PSLIFCO(p,n,ntuple(zz,4)...,ntuple(ff,2)...,ntuple(zz,4))
end

# connection 
# must keep track of conductances 

struct ConnLIFCO_fixed{NP_post,NP_pre} <: Connection
  neurontype_post::NP_post
  weights::SparseMatrixCSC{Float64,Int64}
  neurontype_pre::NP_pre
  post_conductance::Vector{Float64}
  function ConnLIFCO_fixed(post::Npo,
      weights::SparseMatrixCSC,pre::Npr,npost::Int64) where {Npo,Npr}
    new{Npo,Npr}(post,weights,pre,zeros(Float64,npost))
  end
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

# examines state_now, if above threshold, isfiring and refractory turn true
# and potential is reset. Finally, removes expired refractoriness
function _spiking_state_update!(state_now::Vector{R},isrefractory::BitArray{1},
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

# what about external Poisson inputs ?

struct NTPoissonCO <: NeuronType
  rate::Ref{Float64} # rate is in Hz and is mutable
  τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
  v_reversal::Float64 # reversal potential that affects postsynaptic neurons
end

struct PSPoisson{NT} <: PopulationState{NT}
  neurontype::NT
  n::Int64 # pop size
	isfiring::BitArray{1} # firing will be i.i.d. Poisson
  isfiring_alloc::Vector{Float64} # allocate probabilities
  function PSPoisson(p::NT,n) where NT<:NeuronType
    new{NT}(p,n,falses(n),zeros(Float64,n))
  end
end

# Note: to have local updates, you should use a ConnLIFCO_fixed from the Poisson
# to the connected population(s), but also include 
# an UnconnectedPopulation with the 
# Poissoin population state in it when you define the network.

function local_update!(t_now::Float64,dt::Float64,ps::PSPoisson)
  c = dt*ps.rate[]
  @assert c < 1 "Frequency of Poisson input too high!"
  rand!(ps.isfiring_alloc)
  map!(r->r < c, ps.isfiring, ps.isfiring_alloc)
  return nothing
end

# forward_signal!(...)  no need to redefine it! 
