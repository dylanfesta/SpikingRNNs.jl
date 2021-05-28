# Hawkes point processes
# I will try to do exact simulations.


struct PSHawkes{P} <: PopulationState
  population::P
  state_now::Vector{Float64}  # here the state is the same as the input
  spike_proposal::Vector{Float64}
  last_update::Vector{Float64}
	isfiring::BitArray{1}
end
function PSHawkes(p::Population)
  PSHawkes(p, [zeros(Float64,p.n) for _ in (1:4) ]... )
end

struct ConnectionHawkes <: Connection
  postps::PSHawkes # post population 
  preps::PSHawkes # pre population 
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
end
function ConnectionRate(post::PSHawkes,weights::SparseMatrixCSC,pre::PSHawkes)
  aR,aC,_ = findnz(weights)
  ConnectionRate(post,pre,aR,aC,weights)
end

function dynamics_step!(t::Real,dt::Float64,ps::PSHawkes)
  @. ps.spike_proposals = hawkes_next_spike(ps.population,state_now)
  (next_t,next_idx) = argmin(ps.spike_proposals)
  # set the neuron on spiked
  reset_spikes!(ps)
  ps[next_idx] = true
  # update current states
  Δt = next_t - ps.last_update[1]
  @. ps.state_now = hawkes_update(state_now,Δt,ps.population)
  ps.last_update[1] = next_t
  # all done!
  return nothing
end

struct PopulationHawkesExp <: Population
  n::Int64 # pop size
  β::Float64 # time constant
end
Base.broadcastable(p::PopulationHawkesExp) = Ref(p) # does not broadcast

function send_signal!(t::Real,conn::ConnectionHawkes)
  # either zero or one neuron fire, nothing more
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	for _pre in findall(conn.preps.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
			# update the state of the neuron
			conn.postps.state_now[post_idx] += weightsnz[_pnz] 
		end
	end
  return nothing
end

function hawkes_next_spike(pop::PopulationHawkesExp,state::Float64;Tmax::Float64=10.)
  ε=eps(100.)
  t = 0. # this is the running time, from state
  while t<Tmax # Tmax is upper limit, if rate too low 
    M = state-ε
    Δt = rand(Exponential(inv(M)))
    t = t+Δt
    u = rand(Uniform(0,M))
    if u <= state*exp(-pop.β*(t)) # exponential kernel
      return t
    end
  end
  return Tmax
end

@inline function hawkes_update(state_old::Real,Δt::Real,population::PopulationHawkesExp)
  return state_old*exp(-pop.β*∇t) # exponential kernel
end