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
			# update the state of the neuron (input used for external current)
			conn.postps.state_now[post_idx] += weightsnz[_pnz] 
		end
	end
  return nothing
end


# No plasticity here, or evolution of any kind
@inline function dynamics_step!(t_now::Real,dt::Real,conn::ConnectionHawkes)
  return nothing
end

# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
function hawkes_next_spike(pop::PopulationHawkesExp,state::Float64,
    extinp::Float64;Tmax::Float64=100.)
  ε=eps(100.)
  t = 0. # this is the running time, from state
  while t<Tmax # Tmax is upper limit, if rate too low 
    M = max(extinp+state-ε,ε)
    Δt = rand(Exponential(inv(M)))
    t = t+Δt
    u = rand(Uniform(0,M))
    if u <= extinp + state*exp(-pop.β*t) # exponential kernel + external input
      return t
    end
  end
  return Tmax
end

@inline function hawkes_update(state_old::Real,Δt::Real,pop::PopulationHawkesExp)
  return state_old*exp(-pop.β*Δt) # exponential interaction kernel decrease
end



## Utilities and theoretical values

function hawkes_get_spiketimes(neu::Integer,
    actv::Vector{Tuple{N,R}}) where {N<:Integer,R<:Real}
  ret=Vector{R}(undef,0)
  _f = function(ac)
    if ac[1] == neu ; push!(ret,ac[2]); end
  end
  foreach(_f,ac)
  return ret
end

# self excitatory, exponential kernel
function hawkes_exp_self_mean(in::R,w_self::R,β::R) where R
  # should be w_self < β to avoid runaway excitation
  @assert β > w_self "wrong parameters, runaway excitation!"
  return in*β / (β-w_self)
end

function hawkes_exp_self_cov(taus::AbstractVector{R},
    in::R,w_self::R,β::R) where R
  λ = hawkes_exp_self_mean(in,w_self,β)
  C = w_self*λ*(2.0β - w_self) / (2.0(β-w_self))
  return @. C*exp(-(β-w_self)*taus)
end

# numerical

"""
   bin_spikes(Y::Vector{R},dt::R,Tmax::R) where R

Given a vector of spiketimes `Y`,
return a vector of counts, with intervals
`0:dt:Tmax`. The idea is that `dt` is larger than some of 
the time intervals in Y 
"""
function bin_spikes(Y::Vector{R},dt::R,Tmax::R) where R
  bins = 0.0:dt:Tmax
  @assert Y[end] <= bins[end] "Tmax is lower than last event time, or dt should be smaller"
  ret = fill(0,length(bins)-1)
  for t in Y
    k = findfirst(b -> b >= t, bins)
    ret[k-1] += 1
  end
  return ret
end


"""
   covariance_self(Y::Vector{R},τmax::R,dt::R,Tmax::R) where R

Given a vector of spiketimes `Y`,
first it bins spike counts with intervals `0:dt:Tmax`,
then it computes the self-covariance density for intervals 
`0:dt:τmax` (with  `dt<<τmax<<Tmax` )

"""
function covariance_self_numerical(Y::Vector{R},
    τmax::R,dt::R,Tmax::R) where R
  Tmax = Tmax+dt-eps()
  binned = bin_spikes(Y,dt,Tmax)
  ndt_tot = length(binned)
  ndt = round(Integer,τmax/dt)
  ret = Vector{Float64}(undef,ndt)
  binned_sh = similar(binned)
  @inbounds @simd for k in 1:ndt
    circshift!(binned_sh,binned,k)
    ret[k] = dot(binned,binned_sh) 
  end
  ret ./= (ndt_tot*dt^2)
  fm = sum(binned) / Tmax # mean frequency
  ret .-= fm^2
  return ret,(dt:dt:ndt*dt+eps())
end

function covariance_density_numerical(Ys::Vector{Vector{R}}, 
    τmax::R,dt::R,Tmax::R) where R
  Tmax = Tmax+dt-eps()
  times = dt:dt:ndt*dt+eps()
  ndt = round(Integer,τmax/dt)
  n = length(Y)
  ret = Tuple{Tuple{Int64,Int64},Vector{R}}[]
  for i in 1:n ,j in i:n
    cov_ret = Vector{R}(undef,ndt)
    binnedi = bin_spikes(Ys[i],dt,Tmax)
    binnedj = bin_spikes(Ys[j],dt,Tmax)
    binnedj_sh = similar(binnedj)
    ndt_tot = length(binnedi)
    @inbounds @simd for k in 1:ndt
      circshift!(binnedj_sh,binnedj,k)
      cov_ret[k] = dot(binnedi,binnedj_sh)
    end
    ret ./= (ndt_tot*dt^2)
    fm = sum(binned) / Tmax # mean frequency
    cov_ret .-= fm^2
    push!(ret,((i,j),cov_ret))
  end
  return times,ret
end


#=




function binprocess!(p::PointProcess)
  @assert p.dt > 0. "Please set the correct dt (now dt = $dt )"
  Tb = binprocess(p.T,p.dt,p.Tmax)
  p.Tbinned = Tb
  return p
end



function meanfreq_num(p::PointProcess)
  return length(p)/p.Tmax
end

# covariance density
function covariance_self(p::PointProcess,tau_max::Real)
  ndt = round(Integer,tau_max/p.dt)
  ndt_tot = length(p.Tbinned)
  ret = zeros(ndt)
  fm = meanfreq_num(p)
  @showprogress for k in 1:ndt
    ret[k] = dot(p.Tbinned, circshift(p.Tbinned,k)) / (ndt_tot*p.dt^2)
  end
  ret .-= fm^2
  return ret
end




@inline function meanfreq(lam::LambdaExpKer)
  return lam.ν / (1 - lam.α / lam.β)
end

function covariance_self(taus::AbstractVector{R},lam::LambdaExpKer{R}) where R
  λ = meanfreq(lam)
  C = lam.α*λ*(2*lam.β-lam.α) / (2*(lam.β-lam.α))
  return @. C*exp(-(lam.β-lam.α)*taus)
end
=#