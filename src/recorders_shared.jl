

abstract type Recorder end

struct RecStateNow{PS<:PopulationState}
  ps::PS
  isdone::Ref{Bool}
  krec::Int64 # record every k timesteps
  nrecords::Int64 # max recorded steps
  k_warmup::Int64 # starts to record only after this 
  idx_save::Vector{Int64} # which neurons to save. empty -> ALL neurons
  times::Vector{Float64}
  state_now::Matrix{Float64}
  function RecStateNow(ps::PS,everyk::Integer,dt::Float64,Tmax::Float64; 
      idx_save::Vector{Int64}=Int64[],t_warmup::Float64=0.0) where PS
    k_warmup = floor(Int64,t_warmup/dt)
    nrecs = floor(Int64,(Tmax-t_warmup)/(everyk*dt))
    times = fill(NaN,nrecs)
    nneus = isempty(idx_save) ? nneurons(ps) : length(idx_save)
    states = fill(NaN,nneus,nrecs)
    return new{PS}(ps,Ref(false),everyk,nrecs,k_warmup,idx_save,times,states)
  end
end

function reset!(rec::RecStateNow)
  fill!(rec.times,NaN)
  fill!(rec.state_now,NaN)
  rec.isdone[]=false
  return nothing
end

function (rec::RecStateNow)(t::Float64,k::Integer,ntw::AbstractNetwork)
  # I assume k starts from 1, which corresponds to t=0
  kless,_rem=divrem((k-1-rec.k_warmup),rec.krec)
  kless += 1 # vector index must start from 1
  if (_rem != 0) || kless<=0 || rec.isdone[]
    return nothing
  end
  if kless > rec.nrecords 
    rec.isdone[]=true
    return nothing
  end
  if isempty(rec.idx_save)
    rec.state_now[:,kless] .= rec.ps.state_now
  else
    rec.state_now[:,kless] .= rec.ps.state_now[rec.idx_save]
  end
  rec.times[kless] = t
  return nothing
end

# counts the number of spikes
# to simplify things I do it for all neurons
struct RecCountSpikes{PS<:PopulationState}
  ps::PS
  k_warmup::Int64 # starts to record only after this 
  spikecount::Vector{Int64} # how many spikes for each neuron?
  function RecCountSpikes(ps::PS,dt::Float64;
      idx_save::Vector{Int64}=Int64[],t_warmup::Float64=0.0) where PS
    nneus = isempty(idx_save) ? nneurons(ps) : length(idx_save)
    k_warmup = floor(Int64,t_warmup/dt)
    spikecount = fill(0,nneus)
    return new{PS}(ps,k_warmup,spikecount)
  end
end

function reset!(rec::RecCountSpikes)
  fill!(rec.spikecount,0)
  return nothing
end

function (rec::RecCountSpikes)(t::Float64,k::Integer,ntw::AbstractNetwork)
  if k<rec.k_warmup
    return nothing
  else
    for i in findall(rec.ps.isfiring)
      rec.spikecount[i] += 1
    end
    return nothing
  end
end

# utiliy function
function get_mean_rates(rec::RecCountSpikes,dt::Float64,Ttot::Float64)
  ΔT = Ttot-(dt*rec.k_warmup)
  return rec.spikecount ./ ΔT
end


struct RecSpikes{PS<:PopulationState}
  ps::PS
  isdone::Ref{Bool}
  nrecords::Int64 # max recorded steps
  k_now::Ref{Int64} # current spike counter
  t_warmup::Float64 # starts to record only after this time 
  idx_save::Vector{Int64} # which neurons to save. empty -> ALL neurons
  spiketimes::Vector{Float64}
  spikeneurons::Vector{Int64}
  function RecSpikes(ps::PS, expected_rate::Float64,Tmax::Float64;
      idx_save::Vector{Int64}=Int64[],t_warmup::Float64=0.0,
      nrecmax::Int64 = Int64(1E6)) where PS
    nneus = isempty(idx_save) ? nneurons(ps) : length(idx_save)
    nrecs = ceil(Integer,nneus * expected_rate * (Tmax-t_warmup))
    @assert nrecs <= nrecmax "Saving data might require too much memory!"
    spiketimes = fill(NaN,nrecs)
    spikeneurons = fill(-1,nrecs)
    return new{PS}(ps,Ref(false),nrecs,Ref(1),t_warmup,idx_save,
      spiketimes,spikeneurons)
  end
end
function reset!(rec::RecSpikes)
  rec.k_now[]=1
  rec.isdone[]=false
  fill!(rec.spikeneurons,-1)
  fill!(rec.spiketimes,NaN)
  return nothing
end
function (rec::RecSpikes)(t::Float64,k::Integer,ntw::AbstractNetwork)
  if t<rec.t_warmup || rec.isdone[]
    return nothing
  end
  if rec.k_now[] > rec.nrecords 
    rec.isdone[]=true
    return nothing
  end
  for i in findall(rec.ps.isfiring)
    if  (isempty(rec.idx_save) || (i in rec.idx_save) ) && (rec.k_now[] < rec.nrecords)
      rec.spiketimes[rec.k_now[]] = t
      rec.spikeneurons[rec.k_now[]] = i
      rec.k_now[] += 1
    end
  end
  return nothing
end

function get_spiketimes_spikeneurons(rec::RecSpikes)
  _idx = @. !isnan(rec.spiketimes)
  return   rec.spiketimes[_idx],rec.spikeneurons[_idx]
end


# adds a pulse at time of spiking
# voltage_traces : rows are neurons, columns are timesteps
function add_fake_spikes!(v_max::R,voltage_times::Vector{R},
    voltage_traces::Matrix{R},
    spiketimes::Vector{R},spikeneurons::Vector{<:Integer}) where R
  c=1
  for (k,t) in enumerate(voltage_times)
    if checkbounds(Bool,spiketimes,c)
      next_spike = spiketimes[c]
      if t > next_spike
        spikeneuron = spikeneurons[c]
        voltage_traces[spikeneuron,k] = v_max
        c+=1
      end
    end
  end
  return nothing
end

function add_fake_spikes!(v_max::Float64,rtrace::RecStateNow,rspk::RecSpikes)
  return add_fake_spikes!(v_max,rtrace.times,rtrace.state_now,
    rspk.spiketimes,rspk.spikeneurons)
end


struct RecWeights
  weights::SparseMatrixCSC{Float64,Int64}
  isdone::Ref{Bool}
  krec::Int64 # record every k timesteps
  nrecords::Int64 # max recorded steps
  k_warmup::Int64 # starts to record only after this 
  idx_save::Vector{CartesianIndex{2}} # which weights to save
  times::Vector{Float64}
  weight_now::Matrix{Float64}
  function RecWeights(conn::Connection,everyk::Integer,dt::Float64,
      Tmax::Float64; 
      idx_save::Vector{CartesianIndex{2}}=CartesianIndex{2}[],
      t_warmup::Float64=0.0)
    if iszero(n_plasticity_rules(conn))
      @warn "The connection has no plasticity, there is no need of tracking it!"
    end
    k_warmup = floor(Int64,t_warmup/dt)
    nrecs = floor(Int64,(Tmax-t_warmup)/(everyk*dt))
    times = fill(NaN,nrecs)
    if isempty(idx_save)
      x,y,_ = findnz(conn.weights)
      idx_save = CartesianIndex.(x,y)
    end
    nweights = length(idx_save)
    weight_now = fill(NaN,nweights,nrecs)
    return new(conn.weights,Ref(false),everyk,nrecs,k_warmup,idx_save,times,weight_now)
  end
end

function (rec::RecWeights)(t::Float64,k::Integer,ntw::AbstractNetwork)
  # I assume k starts from 1, which corresponds to t=0
  kless,_rem=divrem((k-1-rec.k_warmup),rec.krec)
  kless += 1 # vector index must start from 1
  if (_rem != 0) || kless<=0 || rec.isdone[]
    return nothing
  end
  if kless > rec.nrecords 
    rec.isdone[]=true
    return nothing
  end
  rec.state_now[:,kless] .= rec.weights[rec.idx_save]
  rec.times[kless] = t
  return nothing
end
