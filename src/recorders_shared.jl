

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
  kless,_rem=divrem((k-rec.k_warmup),rec.krec)
  kless += 1 # start from index 0
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
    nrecs = Int64(nneus * expected_rate * (Tmax-t_warmup))
    @assert nrecs <= nrecmax "Saving data might require too much memory!"
    spiketimes = fill(NaN,nrecs)
    spikeneurons = fill(-1,nrecs)
    return new{PS}(ps,Ref(false),nrecs,Ref(1),t_warmup,idx_save,
      spiketimes,spikeneurons)
  end
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
    if  ( isempty(rec.idx_save) || (i in rec.idx_save) ) && (rec.k_now[] < rec.nrecords)
      spiketimes[k_now] = t
      spikeneurons[k_now] = i
      rec.k_now[] += 1
    end
  end
  return nothing
end