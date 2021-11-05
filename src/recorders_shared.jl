

abstract type Recorder end

struct RecStateNow{PS<:PopulationState}
  ps::PS
  isdone::Ref{Bool}
  krec::Int64 # record every k timesteps
  nrecords::Int64 # max recorded steps
  k_start::Int64 # starts to record only after this 
  idx_save::Vector{Int64} # which neurons to save. empty -> ALL neurons
  times::Vector{Float64}
  state_now::Matrix{Float64}
  function RecStateNow(ps::PS,everyk::Integer,dt::Float64,Tend::Float64; 
      idx_save::Vector{Int64}=Int64[],Tstart::Float64=0.0) where PS
    k_start = floor(Int64,Tstart/dt)
    nrecs = floor(Int64,(Tend-Tstart)/(everyk*dt))
    times = fill(NaN,nrecs)
    nneus = isempty(idx_save) ? nneurons(ps) : length(idx_save)
    states = fill(NaN,nneus,nrecs)
    return new{PS}(ps,Ref(false),everyk,nrecs,k_start,idx_save,times,states)
  end
end

function reset!(rec::RecStateNow)
  fill!(rec.times,NaN)
  fill!(rec.state_now,NaN)
  rec.isdone[]=false
  return nothing
end

function (rec::RecStateNow)(t::Float64,k::Integer,::AbstractNetwork)
  # I assume k starts from 1, which corresponds to t=0
  kless,_rem=divrem((k-1-rec.k_start),rec.krec)
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
  Tstart::Float64 # starts to record only after this time 
  Tend::Float64  # max recorded time
  idx_save::Vector{Int64} # which neurons to save. empty -> ALL neurons
  spiketimes::Vector{Float64}
  spikeneurons::Vector{Int64}
  function RecSpikes(ps::PS, expected_rate::Float64,Tend::Float64;
      idx_save::Vector{Int64}=Int64[],Tstart::Float64=0.0,
      nrecmax::Int64 = 10_000_000) where PS
    @assert (isempty(idx_save)  || issorted(idx_save)) "Plase store indexes to save in sorted order"
    nneus = isempty(idx_save) ? nneurons(ps) : length(idx_save)
    if nneus == length(idx_save)  # if 1:Ntot, no need to keep it
      idx_save = Int64[]
    end
    nrecs = ceil(Integer,nneus * expected_rate * (Tend-Tstart))
    @assert nrecs <= nrecmax "Saving data might require too much memory!"
    spiketimes = fill(NaN,nrecs)
    spikeneurons = fill(-1,nrecs)
    return new{PS}(ps,Ref(false),nrecs,Ref(1),Tstart,Tend,idx_save,
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

# little helper function
function _in_sorted(v::Vector{I},el::I) where I
  return !isempty(searchsorted(v,el))
end

function (rec::RecSpikes)(t::Float64,::Integer,::AbstractNetwork)
  if t < rec.Tstart || rec.isdone[]
    return nothing
  end
  if (rec.k_now[] > rec.nrecords) || (t > rec.Tend ) 
    rec.isdone[]=true
    return nothing
  end
  read_all_neus = isempty(rec.idx_save)
  for (neu,isfiring) in enumerate(rec.ps.isfiring)
    if isfiring && (read_all_neus || _in_sorted(rec.idx_save,neu))
      know = rec.k_now[]
      rec.spiketimes[know] = t
      rec.spikeneurons[know] = neu
      rec.k_now[] = know+1
    end
  end
  return nothing
end

function get_spiketimes_spikeneurons(rec::RecSpikes)
  _idx = @. !isnan(rec.spiketimes)
  return   rec.spiketimes[_idx],rec.spikeneurons[_idx]
end
function get_spiketimes_dictionary(rec::RecSpikes)
  spk_t,spk_neu = get_spiketimes_spikeneurons(rec)
  ret=Dict{Int64,Vector{Float64}}()
  for neu in unique(spk_neu)
    idx = spk_neu.==neu
    ret[neu] = spk_t[idx]
  end
  return ret
end

function get_mean_rates(rec::RecSpikes,dt::Float64,Ttot::Float64)
  ΔT = Ttot-rec.Tstart
  dict = get_spiketimes_dictionary(rec)
  ret = Dict{Int64,Float64}()
  for (neu,spks) in pairs(dict)
    ret[neu] = length(spks)/ΔT
  end
  return ret
end

# adds a pulse at time of spiking
# voltage_traces : rows are neurons, columns are timesteps
function add_fake_spikes!(v_spike::R,
    voltage_times::Vector{R},
    voltage_traces::Matrix{R},
    spiketimes::Vector{R},
    spikeneurons::Vector{I},
    idx_save::Vector{I}) where {R,I}
  nneus,nvtimes = size(voltage_traces)
  if isempty(idx_save)
    idx_save = collect(1:nneus)
  end
  for i in 1:nneus
    idx_neu = idx_save[i]
    idx_tspike = findall(==(idx_neu),spikeneurons)
    for k in idx_tspike
      tspike = spiketimes[k]
      idx_vtime = searchsortedfirst(voltage_times,tspike)
      if idx_vtime <= nvtimes
        voltage_traces[i,idx_vtime] = v_spike
      end
    end
  end
  return nothing
end

function add_fake_spikes_old!(v_spike::R,voltage_times::Vector{R},
    voltage_traces::Matrix{R},
    spiketimes::Vector{R},
    spikeneurons::Vector{I},
    idx_save::Vector{I}) where {R,I}
  c=1
  if isempty(idx_save)
    idx_save = collect(1:size(voltage_traces,1))
  end
  for (k,t) in enumerate(voltage_times)
    if checkbounds(Bool,spiketimes,c)
      next_spike = spiketimes[c]
      if t >= next_spike
        idx_neu = findfirst(==(spikeneurons[c]),idx_save)
        if !isnothing(idx_neu)
          voltage_traces[idx_neu,k] = v_spike
        end
        c+=1
      end
    end
  end
  return nothing
end



function add_fake_spikes!(v_spike::Float64,rtrace::RecStateNow,rspk::RecSpikes)
  return add_fake_spikes!(v_spike,rtrace.times,rtrace.state_now,
    rspk.spiketimes,rspk.spikeneurons,rtrace.idx_save)
end

##

function binned_spikecount(dt::Float64,rspk::RecSpikes;
    Nneurons::Int64=-1,Ttot::Float64=0.0)
  return binned_spikecount(dt,get_spiketimes_spikeneurons(rspk)...;
    Nneurons=Nneurons,Ttot=Ttot)
end
function binned_spikecount(dt::Float64,spktimes::Vector{Float64},
    spkneurons::Vector{Int64};Nneurons::Int64=-1,Ttot::Float64=0.0)
  Ttot = max(Ttot, maximum(spktimes)+dt)
  Nneus = max(Nneurons,maximum(spkneurons))
  tbins = 0.0:dt:Ttot
  ntimes = length(tbins)-1
  binnedcount = fill(0,(Nneus,ntimes))
  for (t,neu) in zip(spktimes,spkneurons)
    if t >= tbins[1] # just in case
      tidx = searchsortedfirst(tbins,t)-1
      binnedcount[neu,tidx]+=1
    end
  end
  binsc=midpoints(tbins)
  return binsc,binnedcount
end

# simular to the above, but averages over selected neurons
# and returns a value in Hz
function get_psth(idxs_neu::AbstractVector{<:Integer},
    dt::Float64,rspk::RecSpikes;
    Nneurons::Int64=-1,Ttot::Float64=0.0)
  return get_psth(idxs_neu,dt,get_spiketimes_spikeneurons(rspk)...;
    Nneurons=Nneurons,Ttot=Ttot)
end
function get_psth(idxs_neu::AbstractVector{<:Integer},dt::Float64,spktimes::Vector{Float64},
  spkneurons::Vector{Int64};Nneurons::Int64=-1,Ttot::Float64=0.0)
 tmid,counts = binned_spikecount(dt,spktimes,spkneurons;Nneurons=Nneurons,Ttot=Ttot)
 ret2 = mean(view(counts,idxs_neu,:);dims=1)[:]
 return tmid,ret2 ./ dt
end


function raster_png(dt::Float64,rspk::RecSpikes ;
    Nneurons::Int64=-1,
    Tend::Float64=0.0,spike_height::Int64=5,
    reorder::Vector{Int64}=Int64[])
  spkt,spkn = get_spiketimes_spikeneurons(rspk)
  if isempty(spkn) && (Nneurons==-1 || Tend==0.0)
    error("""
     No spikes recorded ! 
    Impossible to determine the image size. 
    Please set the parameters `Nneurons` and `Tend`
    """)
  end
  Tend = let maxt =  isempty(spkt) ? Tend : maximum(spkt)+dt
    max(Tend, maxt)
  end
  Nneurons = let maxn = isempty(spkn) ? Nneurons : maximum(spkn)
     max(Nneurons,maxn)
  end
  tbins = rspk.Tstart-eps(rspk.Tstart):dt:Tend
  ntimes = length(tbins)-1
  rasterbin = falses(Nneurons,ntimes)
  for (neu,t) in zip(spkn,spkt)
    if t >= tbins[1] # just in case
      tidx = searchsortedfirst(tbins,t)-1
      rasterbin[neu,tidx]=true
    end
  end
  if !isempty(reorder)
    rasterbin .= rasterbin[reorder,:]
  end
  ret = @. RGB(Gray(!rasterbin))
  if spike_height > 1
    ret = repeat(ret;inner=(spike_height,1))
  end
  return ret
end


#=

function raster_png(sd::SpontaneousData,session_id::String;
    cells::Union{Nothing,Vector{Int64}}=nothing,
    spike_height::Int64=3, df_rank::Union{Nothing,DataFrame}=nothing)
  if isnothing(cells)
    dat = filter(r->r.session_id==session_id,sd.cells)
  else
    dat = filter(r->(r.session_id==session_id) && (r.cell in cells),sd.cells)
  end
  if isnothing(df_rank)
    sort!(dat,:cell)
  else
    dat = innerjoin(dat,select(df_rank,:cell,:rank); on=:cell)
    sort!(dat,:rank)
  end
  ret = Vector{Vector{RGB}}(undef,0)
  for r in eachrow(dat)
    # get rasters of corresponding cells
    rast = @. RGB(Gray(! r.raster))
    push!(ret,rast)
  end
  ret = permutedims(hcat(ret...))
  if spike_height > 1
    ret = repeat(ret;inner=(spike_height,1))
  end
  return ret
end
=#

# record weights

struct RecWeights
  weights::SparseMatrixCSC{Float64,Int64}
  isdone::Ref{Bool}
  krec::Int64 # record every k timesteps
  nrecords::Int64 # max recorded steps
  k_warmup::Int64 # starts to record only after this 
  idx_save::Vector{CartesianIndex{2}} # which weights to save
  times::Vector{Float64}
  weights_now::Matrix{Float64}
  function RecWeights(conn::Connection,
      everyk::Integer,dt::Float64,Tmax::Float64; 
      idx_save::Vector{CartesianIndex{2}}=CartesianIndex{2}[],
      Tstart::Float64=0.0)
    if iszero(n_plasticity_rules(conn))
      @warn "The connection has no plasticity, there is no need of tracking it!"
    end
    k_warmup = floor(Int64,Tstart/dt)
    nrecs = floor(Int64,(Tmax-Tstart)/(everyk*dt))
    times = fill(NaN,nrecs)
    if isempty(idx_save)
      x,y,_ = findnz(conn.weights)
      idx_save = CartesianIndex.(x,y)
    end
    nweights = length(idx_save)
    weights_now = fill(NaN,nweights,nrecs)
    return new(conn.weights,Ref(false),everyk,nrecs,k_warmup,idx_save,times,weights_now)
  end
end

Base.length(rec::RecWeights) = length(rec.times)

function reset!(rec::RecWeights)
  fill!(rec.times,NaN)
  fill!(rec.weights_now,NaN)
  rec.isdone[]=false
  return nothing
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
  rec.weights_now[:,kless] .= rec.weights[rec.idx_save]
  rec.times[kless] = t
  return nothing
end

function rebuild_weights(rec::RecWeights)
  times = rec.times
  ntimes = length(times)
  wout = Vector{SparseMatrixCSC{Float64,Int64}}(undef,ntimes)
  for kt in 1:ntimes
    _w = zeros(Float64,size(rec.weights))
    for (k,ij) in enumerate(rec.idx_save)
      _w[ij] = rec.weights_now[k,kt]
    end
    wout[kt] = sparse(_w)
  end
  return times,wout
end

# records the full weight matrix
# this makes life much easier where there is structural plasticity.
# BUT please use it with a large k (e.g. 2 Hz )

struct RecWeightsFull
  weights::SparseMatrixCSC{Float64,Int64}
  isdone::Ref{Bool}
  krec::Int64 # record every k timesteps
  nrecords::Int64 # max recorded steps
  k_start::Int64 # starts to record only after this 
  times::Vector{Float64}
  weights_now:: Vector{SparseMatrixCSC{Float64,Int64}}
  function RecWeightsFull(conn::Connection,
      everyk::Integer,dt::Float64,Tend::Float64; 
      Tstart::Float64=0.0)
    if iszero(n_plasticity_rules(conn))
      @warn "The connection has no plasticity, there is no need of tracking it!"
    end
    k_start = floor(Int64,Tstart/dt)
    nrecs = floor(Int64,(Tend-Tstart)/(everyk*dt))
    times = fill(NaN,nrecs)
    weights_now = Vector{SparseMatrixCSC{Float64,Int64}}(undef,nrecs)
    return new(conn.weights,Ref(false),everyk,nrecs,k_start,times,weights_now)
  end
end

Base.length(rec::RecWeightsFull) = length(rec.times)

function reset!(rec::RecWeightsFull)
  fill!(rec.times,NaN)
  fill!(rec.weights_now,NaN)
  rec.isdone[]=false
  return nothing
end

function (rec::RecWeightsFull)(t::Float64,k::Integer,::AbstractNetwork)
  # I assume k starts from 1, which corresponds to t=0
  kless,_rem=divrem((k-rec.k_start-1),rec.krec)
  kless += 1 # vector index must start from 1
  if (_rem != 0) || kless<=0 || rec.isdone[]
    return nothing
  end
  if kless > rec.nrecords 
    rec.isdone[]=true
    return nothing
  end
  rec.weights_now[kless] = copy(rec.weights)
  rec.times[kless] = t
  return nothing
end
