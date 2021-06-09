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
  @. ps.state_now = hawkes_evolve_state(ps.state_now,Δt_next,ps.population)
  # advance time
  ps.time_now[1] += Δt_next
  return nothing
end

# initialization: send inputs at t=0, and have some initial state, if needed
@inline function hawkes_initialize!(ntw::RecurrentNetwork;state_now::Real=1E-2)
  for ps in ntw.population_states
    fill!(ps.state_now,state_now)
    fill!(ps.input,0.0)
    fill!(ps.spike_proposals,Inf)
  end
  send_signal!.(NaN,ntw.inputs)
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
  β = conn.preps.population.β # this is:  interaction_kernel(0,conn.preps.population) 
	for _pre in findall(conn.preps.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
			# update the state of the neuron (input used for external current)
			conn.postps.state_now[post_idx] += weightsnz[_pnz]*β
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
    if u <= extinp + hawkes_evolve_state(state,t,pop) # exponential kernel + external input
      return t
    end
  end
  return Tmax
end

@inline function interaction_kernel(t::Real,pop::PopulationHawkesExp)
  return pop.β*exp(-pop.β*t) # exponential interaction kernel decrease
end
@inline function hawkes_evolve_state(state::Real,Δt::Real,pop::PopulationHawkesExp)
  return state * exp(-pop.β*Δt)  # the beta factor is already in state 
end

@inline function fou_interaction_kernel(ω::Real,pop::PopulationHawkesExp)
  return pop.β/(pop.β - im*sqrt(2π)*ω)
end

## Utilities and theoretical values

function hawkes_get_spiketimes(neu::Integer,actv::Vector{Tuple{N,R}}) where {N<:Integer,R<:Real}
  ret=Vector{R}(undef,0)
  _f = function(ac)
    if ac[1] == neu ; push!(ret,ac[2]); end
  end
  foreach(_f,actv)
  return ret
end

# self excitatory, exponential kernel
function hawkes_exp_self_mean(in::R,w_self::R,β::R) where R
  # should be w_self < β to avoid runaway excitation
  # @assert β > w_self "wrong parameters, runaway excitation!"
  return in / (1-w_self)
  # return in / (β-w_self)
end

# eq (16) in Hawkes 1971b
function hawkes_exp_self_cov(taus::AbstractVector{R},
    in::R,w_self::R,β::R) where R
  λ = hawkes_exp_self_mean(in,w_self,β)
  C = w_self*λ*(2.0 - w_self) / (2.0 (1-w_self))
  return @. C*exp(-(1-w_self)*taus)
end

# numerical

"""
   bin_spikes(Y::Vector{R},dt::R,Tmax::R) where R

Given a vector of spiketimes `Y`,
return a vector of counts, with intervals
`0:dt:Tmax`. So that count length is Tmax/dt.
Ignores event times larger than Tmax 
"""
function bin_spikes(Y::Vector{R},dt::R,Tmax::R) where R
  bins = 0.0:dt:Tmax
  #@assert Y[end] <= bins[end] "Tmax is lower than last event time, or dt should be smaller"
  ret = fill(0,length(bins)-1)
  for t in Y
    k::Int64 = something(findfirst(b -> b >= t, bins),0)
    if k > 1
      ret[k-1] += 1
    end
  end
  return ret
end


# time starts at 0, ends at T-dt, there are T/dt steps in total
@inline function get_times(dt::Real,T::Real)
  return (0.0:dt:(T-dt))
end

# frequencies for Fourier transform. In total T/dt * 2 
function get_frequencies(dt::Real,T::Real)
  dω = inv(2T)
  ωmax = inv(2dt)
  ret = collect(-ωmax:dω:(ωmax-dω))
  (val,idx) = findmin(abs.(ret))
  if abs(val)<1E-10
    ret[idx] = 1E-9
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
function covariance_self_numerical(Y::Vector{R},dτ::R,τmax::R,
    Tmax::Union{R,Nothing}=nothing) where R
  Tmax = something(Tmax,Y[end]-dτ)
  binned = bin_spikes(Y,dτ,Tmax)
  ndt_tot = length(binned)
  ndt = round(Integer,τmax/dτ)
  ret = Vector{Float64}(undef,ndt)
  binned_sh = similar(binned)
  @inbounds @simd for k in 0:ndt-1
    circshift!(binned_sh,binned,k)
    ret[k+1] = dot(binned,binned_sh) 
  end
  fm = sum(binned) / Tmax # mean frequency
  @. ret = ret /  (ndt_tot*dτ^2) - fm^2
  return ret
end


function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R,
   Tmax::Union{R,Nothing}=nothing ; verbose::Bool=false) where R
  Tmax = something(Tmax, maximum(x->x[end],Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Tuple{Tuple{Int64,Int64},Vector{R}}[]
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tmax/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tmax)
    fmi = length(Ys[i]) / Tmax # mean frequency
    ndt_tot = length(binnedi)
    for j in i:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      cov_ret = Vector{R}(undef,ndt)
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tmax)
      fmj = length(Ys[j]) / Tmax # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        cov_ret[k+1] = dot(binnedi,binnedj_sh)
      end
      @. cov_ret = cov_ret / (ndt_tot*dτ^2) - fmi*fmj
      push!(ret,((i,j),cov_ret))
    end
  end
  return ret
end


function covariance_density_numerical_unnormalized(Ys::Vector{Vector{R}},dτ::Real,τmax::R,
   Tmax::Union{R,Nothing}=nothing ; verbose::Bool=false) where R
  Tmax = something(Tmax, maximum(x->x[end],Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Tuple{Tuple{Int64,Int64},Vector{R}}[]
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tmax/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tmax)
    ndt_tot = length(binnedi)
    for j in i:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      cov_ret = Vector{R}(undef,ndt)
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tmax)
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        cov_ret[k+1] = dot(binnedi,binnedj_sh)
      end
      @. cov_ret = cov_ret / (ndt_tot*dτ^2)
      push!(ret,((i,j),cov_ret))
    end
  end
  return ret
end