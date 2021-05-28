# Hawkes point processes
# I will try to do exact simulations.

struct PSHawkes{P} <: PopulationState
  population::P
  state_now::Vector{Float64} 
  input::Vector{Float64} # input only used for external currents 
  spike_proposals::Vector{Float64}
  time_now::Vector{Float64}
	isfiring::BitArray{1}
end
function PSHawkes(p::Population)
  PSHawkes(p,zeros.(Float64,(p.n,p.n,p.n,1))..., BitArray{1}(undef,p.n))
end

struct ConnectionHawkes <: Connection
  postps::PSHawkes # post population 
  preps::PSHawkes # pre population 
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
end
function ConnectionHawkes(post::PSHawkes,weights::SparseMatrixCSC,pre::PSHawkes)
  aR,aC,_ = findnz(weights)
  ConnectionHawkes(post,pre,aR,aC,weights)
end

function dynamics_step!(t::Real,dt::Float64,ps::PSHawkes)
  @. ps.spike_proposals = hawkes_next_spike(ps.population,ps.state_now,ps.input)
  (Δt_next,next_idx) = findmin(ps.spike_proposals)
  # set the neuron on spiked
  reset_spikes!(ps)
  ps.isfiring[next_idx] = true
  # update current states
  @. ps.state_now = hawkes_update(ps.state_now,Δt_next,ps.population)
  # advance time
  ps.time_now[1] += Δt_next
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
			# update the input of the neuron
			conn.postps.input[post_idx] += weightsnz[_pnz] 
		end
	end
  return nothing
end


# No plasticity here, or evolution of any kind
@inline function dynamics_step!(t_now::Real,dt::Real,conn::ConnectionHawkes)
  return nothing
end

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
function hawkes_next_spike(pop::PopulationHawkesExp,state::Float64,extinp::Float64;Tmax::Float64=100.)
  ε=eps(100.)
  t = 0. # this is the running time, from state
  while t<Tmax # Tmax is upper limit, if rate too low 
    M = max(extinp+state-ε,ε)
    Δt = rand(Exponential(inv(M)))
    t = t+Δt
    u = rand(Uniform(0,M))
    if u <= extinp + state*exp(-pop.β*(t)) # exponential kernel + external input
      return t
    end
  end
  return Tmax
end

@inline function hawkes_update(state_old::Real,Δt::Real,pop::PopulationHawkesExp)
  return state_old*exp(-pop.β*Δt) # exponential interaction kernel decrease
end
