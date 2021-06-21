using Base: Float64
# ./src/lif_conductance.jl

############
# conductance-based (classic) LIF

# threshold-linear input-output function
struct PopLIFCO <: Population
  n::Int64 # pop size
  τ::Float64 # time constant
  Cap::Float64 # capacitance
  v_threshold::Vector{Float64} # spiking threshold (can vary among neurons)
	v_reset::Float64 # reset after spike
  v_leak::Float64 # reversal potential for leak term
	τ_refractory::Float64 # refractory time
	τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
	v_reversal::Float64 # reversal potential that affects postsynaptic neurons
end

struct PSLIFCO{P} <: PopulationState
  population::P
  state_now::Vector{Float64}
  input::Vector{Float64}
	alloc_dv::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
	pre_reverse_potentials::Vector{Float64}
	pre_conductances_now::Vector{Float64}
end
function PSLIFCO(p::Population)
  zz = _ -> zeros(Float64,p.n)
  ff = _-> falses(p.n)
  PSLIF(p, ntuple(zz,4)...,ntuple(ff,2)...,ntuple(zz,4))
end

# Input: dummy neurons with poisson firing and conductance signal 
# for simplicity, define it directly as PopulationState and skip the Population struct
struct PSPoissonCO{P} <: PopulationState
  n::Int64
  rate::Ref{Float64} # rate is in Hz
	τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
	v_reversal::Float64 # reversal potential that affects postsynaptic neurons
  population::P
	isfiring::BitArray{1} # firing will be i.i.d. Poisson
  isfiring_alloc::Vector{Float64} # allocate probabilities
end
function change_rate!(ps::PSPoissonCO,ν::Float64)
  ps.population.rate[]=ν
  return nothing
end


# connection between two conductance populations

struct ConnectionLIFCO <: Connection
  postps::PSLIFCO # postsynaptic population state
  preps::PSLIFCO # presynaptic population state
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
  post_conductance::Vector{Float64}
end
function ConnectionLIFCO(post::PSLIFCO,weights::SparseMatrixCSC,pre::PSLIFCO)
  aR,aC,_ = findnz(weights)
	npost = post.population.n
  ConnectionLIFCO(post,pre,aR,aC,weights,zeros(Float64,npost))
end

struct ConnectionPoisson2LIFCO <: Connection
  postps::PSLIFCO # postsynaptic population state
  preps::PSPoissonCO
  weight::Float64 # I am assuming that ALL neurons receive input ! Later if needed I can add a mask.
  post_conductance::Vector{Float64}
end
function ConnectionLIFCO(post::PSLIFCO,weight::Float64,pre::PSPoissonCO)
  npost = post.population.n
  return ConnectionPoisson2LIFCO(post,pre,weight,zeros(Float64,npost))
end

function dynamics_step!(t_now::Float64,dt::Float64,ps::PSLIFCO)
	# computes the update to internal voltage, given the total input
  # dv =  (v_leak - v ) dt / τ + I dt / Cap
  dttau =  dt / ps.population.τ
  dtCap = dt / ps.population.Cap
  @. ps.alloc_dv = (ps.population.v_leak - ps.state_now)*dttau + ps.input*dtCap
  ps.state_now .+= ps.alloc_dv # v_{t+1} = v_t + dv
	# now spikes and refractoriness
	reset_spikes!(ps)
	@inbounds @simd for i in eachindex(ps.state_now)
		if ps.state_now[i] > ps.population.v_threshold
			ps.isfiring[i] = true
			ps.state_now[i] =  ps.population.v_reset
			ps.last_fired[i] = t_now
			ps.isrefractory[i] = true
		# check only when refractory
		elseif ps.isrefractory[i] && 
				( (t_now-ps.last_fired[i]) >= ps.population.τ_refractory)
			ps.isrefractory[i] = false
		end
	end
	# 
  return nothing
end

function dynamics_step!(t_now::Real,dt::Real,ps::PSPoissonCO)
  # regenerate isfiring (reset is included)
  rand!(ps.isfiring_alloc)
  c = dt*ps.rate
  @assert c<1 "Frequency too high!"
  ps.isfiring .= ps.isfiring_alloc .< c
  return nothing
end


function send_signal!(t_now::Float64,conn::ConnectionLIFCO)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	τ_decay = conn.preps.population.τ_post_conductance_decay
	for _pre in findall(conn.preps.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
		  conn.post_conductance[post_idx] += 
						weightsnz[_pnz] / τ_decay
		end
	end
	# finally , add the currents to postsynaptic input
	# ONLY non-refractory neurons
	post_refr = findall(conn.postps.isrefractory) # refractory ones
	@inbounds @simd for i in eachindex(conn.postps.input)
		if !(i in post_refr)
      # input = conductance(t) * ( v_reversal - v(t) )
			conn.postps.input[i] += 
        conn.post_conductance[i] * (conn.preps.population.v_reversal - conn.postps.state_now[i])
		end
	end
  return nothing
end

# Signal from Poisson dummy neurons to LIF-conductance neurons
function send_signal!(t_now::Float64,conn::ConnectionPoisson2LIFCO)
  wtau = conn.weight / conn.preps.τ_post_conductance_decay
  for k in findall(conn.preps.isfiring) # findall seems faster than direct indexing, when things are sparse
    conn.post_conductance[k] += wtau
	end
	# add the currents to postsynaptic input , ONLY non-refractory neurons
	@inbounds @simd for i in findall(.! conn.postps.isrefractory)
    conn.postps.input[i] += 
        conn.post_conductance[i] * (conn.preps.v_reversal - conn.postps.state_now[i])
	end
  return nothing
end

# postsynaptic conductances decay in time
function dynamics_step!(t_now::Real,dt::Real,conn::ConnectionLIFCO)
	τ_decay = conn.preps.population.τ_post_conductance_decay
	@inbounds @simd for i in eachindex(conn.post_conductance)
		conn.post_conductance[i] -= dt*conn.post_conductance[i] / τ_decay
	end
	return nothing
end
# same for dummy neurons
function dynamics_step!(t_now::Real,dt::Real,conn::ConnectionPoisson2LIFCO)
	τ_decay = conn.preps.τ_post_conductance_decay
	@inbounds @simd for i in eachindex(conn.post_conductance)
		conn.post_conductance[i] -= dt*conn.post_conductance[i] / τ_decay
	end
	return nothing
end


# static constant input to LIFCO population
@inline function send_signal!(t::Real,input::PopInputStatic{P}) where P<:PSLIFCO
	refr = findall(input.population_state.isrefractory) # refractory ones
  @inbounds @simd for i in eachindex(input.population_state.input)
		if !(i in refr)
  		input.population_state.input[i] += input.h[i]
		end
	end
  return nothing
end

