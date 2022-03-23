

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
    isdone = Ref(false)
    if nrecs < 1 
      @warn "something wrong in the recorder... I assume this is a test ?"
      nrecs=1
      isdone = Ref(true)
    end
    times = fill(NaN,nrecs)
    nneus = isempty(idx_save) ? nneurons(ps) : length(idx_save)
    states = fill(NaN,nneus,nrecs)
    return new{PS}(ps,isdone,everyk,nrecs,k_start,idx_save,times,states)
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
  spiketimes::Vector{Float32} # resolution 1E-7 , still below dt
  spikeneurons::Vector{UInt64} # max is about 60_000 neurons, which is plenty
  function RecSpikes(ps::PS, expected_rate::Float64,Tend::Float64;
      idx_save::Vector{Int64}=Int64[],Tstart::Float64=0.0,
      nrecmax::Int64 = 10_000_000) where PS
    @assert (isempty(idx_save)  || issorted(idx_save)) "Plase store indexes to save in sorted order"
    nneus = isempty(idx_save) ? nneurons(ps) : length(idx_save)
    if nneus == nneurons(ps)  # if 1:Ntot, no need to keep it
      idx_save = Int64[]
    end
    nrecs = ceil(Integer,nneus * expected_rate * (Tend-Tstart))
    isdone = Ref(false)
    if nrecs < 1 
      @warn "something wrong in the recorder... I assume this is a test ?"
      nrecs=1
      isdone = Ref(true)
    end
    @assert nrecs <= nrecmax "Saving data might require too much memory!"
    spiketimes = fill(NaN32,nrecs)
    spikeneurons = fill(UInt16(0),nrecs) # no neuron should be 0
    return new{PS}(ps,isdone,nrecs,Ref(1),Tstart,Tend,idx_save,
      spiketimes,spikeneurons)
  end
end
function reset!(rec::RecSpikes)
  rec.k_now[]=1
  rec.isdone[]=false
  fill!(rec.spikeneurons,UInt16(0))
  fill!(rec.spiketimes,NaN32)
  return nothing
end

# easy to save wrapper
struct RecSpikesContent
  Tstart::Float64 # starts to record only after this time 
  Tend::Float64  # max recorded time
  idx_save::Vector{Int64} # which neurons to save. empty -> ALL neurons
  spiketimes::Vector{Float32} # resolution 1E-7 , still below dt
  spikeneurons::Vector{UInt64} # max is about 60_000 neurons, which is plenty
  function RecSpikesContent(r::RecSpikes)
    to_keep = findall(isfinite,r.spiketimes) 
    new(r.Tstart,r.Tend,r.idx_save,r.spiketimes[to_keep],r.spikeneurons[to_keep])
  end
end
function get_content(rec::RecSpikes)
  return RecSpikesContent(rec)
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
      rec.spiketimes[know] = Float32(t)
      rec.spikeneurons[know] = UInt16(neu)
      rec.k_now[] = know+1
      if know+1 > rec.nrecords # must check for each spike
        rec.isdone[]=true
        return nothing
      end
    end
  end
  return nothing
end

function get_spiketimes_spikeneurons(rec::Union{RecSpikes,RecSpikesContent})
  _idx = findall(isfinite,rec.spiketimes)
  return Float64.(rec.spiketimes[_idx]),Int64.(rec.spikeneurons[_idx])
end

function get_spiketimes_dictionary(rec::Union{RecSpikes,RecSpikesContent})
  spk_t,spk_neu = get_spiketimes_spikeneurons(rec)
  ret=Dict{Int64,Vector{Float64}}()
  for neu in unique(spk_neu)
    idx = spk_neu.==neu
    ret[neu] = spk_t[idx]
  end
  return ret
end

# does not necessarily preserve neuron information, but it is useful for rasters
function get_spiketrains(rec::Union{RecSpikes,RecSpikesContent};resort=nothing)
  spk_t,spk_neu = get_spiketimes_spikeneurons(rec)
  neus = sort!(unique(spk_neu))
  N = maximum(neus)
  ret = map(1:N) do neu
    if (neu in neus)
      idxs = spk_neu .== neu
      return spk_t[idxs]
    else
      return Float64[]
    end
  end
  return ret
end

function get_mean_rates(rec::Union{RecSpikes,RecSpikesContent};
    (Tend::Float64)=0.0)
  Tend = min(Tend,rec.Tend)  
  ΔT = Tend-rec.Tstart
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

function add_fake_spikes!(v_spike::Float64,rtrace::RecStateNow,rspk::Union{RecSpikes,RecSpikesContent})
  spiketimes,spikeneurons = get_spiketimes_spikeneurons(rspk)
  return add_fake_spikes!(v_spike,rtrace.times,rtrace.state_now,
    spiketimes,spikeneurons,rtrace.idx_save)
end

##

function binned_spikecount(dt::Float64,rspk::Union{RecSpikes,RecSpikesContent};
    Nneurons::Int64=-1,Tend::Float64=-1.0)
  return binned_spikecount(dt,get_spiketimes_spikeneurons(rspk)...;
    Nneurons=Nneurons,Tend=max(Tend,rspk.Tend),Tstart=rspk.Tstart)
end
function binned_spikecount(dt::Float64,spktimes::Vector{Float64},
    spkneurons::Vector{Int64};Nneurons::Int64=-1,
      Tend::Float64=-1.0,Tstart::Float64=0.0)
  Tend = max(Tend, maximum(spktimes)+dt)
  Nneus = max(Nneurons,maximum(spkneurons))
  tbins = Tstart:dt:Tend
  ntimes = length(tbins)-1
  binnedcount = fill(0,(Nneus,ntimes))
  for (t,neu) in zip(spktimes,spkneurons)
    if (tbins[1] < t <= tbins[end]) # just in case
      tidx = searchsortedfirst(tbins,t)-1
      binnedcount[neu,tidx]+=1
    end
  end
  binsc=midpoints(tbins)
  return binsc,binnedcount
end


# convolve spiketimes with negative exponential
# for SINGLE / GROUP of neurons (not all)
function get_spike_exp_convolution(dt::R,tau_exp::R,
    spiktimes::Vector{R},spkneurons::Vector{I},
    Nneurons::I,Tstart::R,Tend::R ; 
    idx_neurons::Union{Vector{I},Nothing}=nothing) where {R,I}
  binnedt,binned_spikes = binned_spikecount(dt,spiktimes,spkneurons;
    Nneurons=Nneurons,Tstart=Tstart,Tend=Tend)
  @assert tau_exp > dt "convolution is too narrow!"  
  convvect = @. (exp(-(0:dt:(10tau_exp))/tau_exp))/tau_exp
  idx_neurons = something(idx_neurons,(1:Nneurons))
  bin_select = sum(view(binned_spikes,idx_neurons,:);dims=1)[:] ./ length(idx_neurons)
  ret = conv(convvect,bin_select)
  return binnedt,ret[1:length(binnedt)]
end

function get_spike_exp_convolution(dt::R,tau_exp::R,
    rspk::Union{RecSpikes,RecSpikesContent},Nneurons::I;
    idx_neurons::Union{Vector{I},Nothing}=nothing) where {R,I}
  return get_spike_exp_convolution(dt,tau_exp,
    get_spiketimes_spikeneurons(rspk)...,Nneurons,
    rspk.Tstart,rspk.Tend;
    idx_neurons=idx_neurons)
end


# simular to the above, but averages over selected neurons
# and returns a value in Hz
function get_psth(idxs_neu::AbstractVector{<:Integer},
    dt::Float64,rspk::Union{RecSpikes,RecSpikesContent};
    Nneurons::Int64=-1,Tend::Float64=-1.0)
  return get_psth(idxs_neu,dt,get_spiketimes_spikeneurons(rspk)...;
    Nneurons=Nneurons,Tend=max(Tend,rspk.Tend),Tstart=rspk.Tstart)
end
function get_psth(idxs_neu::AbstractVector{<:Integer},dt::Float64,spktimes::Vector{Float64},
  spkneurons::Vector{Int64};Nneurons::Int64=-1,Tend::Float64=-1.0,
  Tstart::Float64=0.0)
 tmid,counts = binned_spikecount(dt,spktimes,spkneurons;
  Nneurons=Nneurons,Tend=Tend,Tstart=Tstart)
 ret2 = mean(view(counts,idxs_neu,:);dims=1)[:]
 return tmid,ret2 ./ dt
end

function raster_png(dt::Float64,rspk::Union{RecSpikes,RecSpikesContent};
    Nneurons::Int64=-1,
    Tend::Float64=0.0,spike_height::Int64=5,
    reorder::Vector{Int64}=Int64[])
  spkt,spkn = get_spiketimes_spikeneurons(rspk)
  if iszero(Tend)
    Tend = max(rspk.Tend,maximum(spkt)+dt)
  end
  if isempty(spkn) && (Nneurons==-1)
    error("""
     No spikes recorded ! 
    Impossible to determine the image size. 
    Please set the parameters `Nneurons` and `Tend`
    """)
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

"""
    bin_spikes(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R

# Arguments
  + `Y::Vector{<:Real}` : vector of spike times
  + `dt::Real` : time bin size
  + `Tend::Real` : end time for the raster
# Optional argument
  + `Tstart::Real=0.0` : start time for the raster

# Returns   
  + `binned_spikes::Vector{<:Integer}` : `binned_spikes[k]` is the number of spikes that occur 
      in the timebin `k`  (i.e. between `Tstart + (k-1)*dt` and `Tstart + k*dt`)
"""
function bin_spikes(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)  
  ret = fill(0,length(times)-1)
  for y in Y
    if Tstart < y <= last(times)
      k = searchsortedfirst(times,y)-1
      ret[k] += 1
    end
  end
  return ret
end


"""
  draw_spike_raster(trains::Vector{Vector{Float64}},
      dt::Real,Tend::Real;
      Tstart::Real=0.0,
      spike_size::Integer = 5,
      spike_separator::Integer = 1,
      background_color::Color=RGB(1.,1.,1.),
      spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
      max_size::Real=1E4) where C<:Color

Draws a matrix that contains the raster plot of the spike train.

# Arguments
  + `Trains` :  Vector of spike trains. The order of the vector corresponds to 
    the order of the plot. First element is at the top, second is second row, etc.
  + `dt` : time interval representing one horizontal pixel  
  + `Tend` : final time to be considered

# Optional arguments
  + `Tstart::Real` : starting time
  + `spike_size::Integer` : heigh of spike (in pixels)
  + `spike_separator::Integer` : space between spikes, and vertical padding
  + `background_color::Color` : self-explanatory
  + `spike_colors::Union{Color,Vector{Color}}` : if a single color, color of all spikes, if vector of colors, 
     color for each neuron (length should be same as number of neurons)
  + `max_size::Integer` : throws an error if image is larger than this number (in pixels)

# Returns
  + `raster_matrix::Matrix{Color}` you can save it as a png file
"""
function draw_spike_raster(trains::Vector{Vector{Float64}},
  dt::Real,Tend::Real;
    Tstart::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where C<:Color
  nneus = length(trains)
  if typeof(spike_colors) <: Color
    spike_colors = repeat([spike_colors,];outer=nneus)
  else
    @assert length(spike_colors) == nneus "error in setting colors"
  end
  binned_binary  = map(trains) do train
    .! iszero.(bin_spikes(train,dt,Tend;Tstart=Tstart))
  end
  ntimes = length(binned_binary[1])
  ret = fill(background_color,
    (nneus*spike_size + # spike sizes
      spike_separator*nneus + # spike separators (incl. top/bottom padding) 
      spike_separator),ntimes)
  @assert all(size(ret) .< max_size ) "The image is too big! Please change time limits"  
  for (neu,binv,col) in zip(1:nneus,binned_binary,spike_colors)
    spk_idx = findall(binv)
    _idx_pre = (neu-1)*(spike_size+spike_separator)+spike_separator
    y_color = _idx_pre+1:_idx_pre+spike_size
    ret[y_color,spk_idx] .= col
  end
  return ret
end


# record weights
# might be useful when testing plasticity at small scale,
# otherwise use RecWeightsFull, below

struct RecWeightsApplyFunction
  weights::SparseMatrixCSC{Float64,Int64}
  isdone::Ref{Bool}
  krec::Int64 # record every k timesteps
  nrecords::Int64 # max recorded steps
  k_start::Int64 # starts to record only after this 
  fun_things::Function # stuff to do, takes SparseMatrix, returns vector
  out_size::Int64 # size of function output vector
  times::Vector{Float64}
  fun_output::Matrix{Float64}
  function RecWeightsApplyFunction(conn::Connection,thefun::Function,
      everyk::Integer,dt::Float64,Tmax::Float64; 
      Tstart::Float64=0.0)
    if iszero(n_plasticity_rules(conn))
      @warn "The connection has no plasticity, there is no need of tracking it!"
    end
    out_sample = thefun(conn.weights)
    typeassert(out_sample,Vector{Float64})
    out_size = length(out_sample)
    k_start = floor(Int64,Tstart/dt)
    nrecs = floor(Int64,(Tmax-Tstart)/(everyk*dt))
    isdone = Ref(false)
    if nrecs < 1 
      @warn "something wrong in the recorder... I assume this is a test ?"
      nrecs=1
      isdone = Ref(true)
    end
    times = fill(NaN,nrecs)
    fun_output = fill(NaN,out_size,nrecs)
    return new(conn.weights,isdone,everyk,nrecs,k_start,
      thefun,out_size,
      times,fun_output)
  end
end
Base.length(rec::RecWeightsApplyFunction) = length(rec.times)

# easy to save wrapper
struct RecWeightsApplyFunctionContent
  times::Vector{Float64}
  fun_output::Matrix{Float64}
  function RecWeightsApplyFunctionContent(r::RecWeightsApplyFunction)
    new(r.times,r.fun_output)
  end
end
function get_content(rec::RecWeightsApplyFunction)
  return RecWeightsApplyFunctionContent(rec)
end


function reset!(rec::RecWeightsApplyFunction)
  fill!(rec.times,NaN)
  fill!(rec.fun_output,NaN)
  rec.isdone[]=false
  return nothing
end

function (rec::RecWeightsApplyFunction)(t::Float64,k::Integer,::AbstractNetwork)
  kless,_rem=divrem((k-1-rec.k_start),rec.krec)
  kless += 1 # vector index must start from 1
  if (_rem != 0) || kless<=0 || rec.isdone[]
    return nothing
  end
  if kless > rec.nrecords 
    rec.isdone[]=true
    return nothing
  end
  rec.fun_output[:,kless] .= rec.fun_things(rec.weights)
  rec.times[kless] = t
  return nothing
end

# function rebuild_weights(rec::RecWeights)
#   times = rec.times
#   ntimes = length(times)
#   wout = Vector{SparseMatrixCSC{Float64,Int64}}(undef,ntimes)
#   for kt in 1:ntimes
#     _w = zeros(Float64,size(rec.weights))
#     for (k,ij) in enumerate(rec.idx_save)
#       _w[ij] = rec.weights_now[k,kt]
#     end
#     wout[kt] = sparse(_w)
#   end
#   return times,wout
# end

# records the full weight matrix
# this makes life much easier where there is structural plasticity.
# BUT please use it with a large k (e.g. 2 Hz )

"""
  RecWeightsFull(conn::Connection,
        everyk::Integer,dt::Float64,Tend::Float64; 
        Tstart::Float64=0.0)

Constructor, fields are self explanatory        
"""
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
    isdone = Ref(false)
    if nrecs < 1 
      @warn "something wrong in the recorder... I assume this is a test ?"
      nrecs=1
      isdone = Ref(true)
    end
    times = fill(NaN,nrecs)
    weights_now = Vector{SparseMatrixCSC{Float64,Int64}}(undef,nrecs)
    return new(conn.weights,isdone,everyk,nrecs,k_start,times,weights_now)
  end
end
Base.length(rec::RecWeightsFull) = length(rec.times)

# easy to save wrapper
struct RecWeightsFullContent
  times::Vector{Float64}
  weights_now::Vector{SparseMatrixCSC{Float64,Int64}}
  function RecWeightsFullContent(r::RecWeightsFull)
    new(r.times,r.weights_now)
  end
end
function get_content(rec::RecWeightsFull)
  return RecWeightsFullContent(rec)
end


function reset!(rec::RecWeightsFull)
  fill!(rec.times,NaN)
  fill!(rec.weights_now,sparse(zeros(1,1)))
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

# utility functions for differences in weights

_wdiff_plain(w1::R,w2::R) where R<:Real = w1 - w2
_wdiff_plain(wmat1::Matrix{R},wmat2::Matrix{R}) where R<:Real = _wdiff_plain.(wmat1,wmat2)
function _wdiff_rel(w1::R,w2::R) where R<:Real
  wavg = 0.5(w1+w2)
  (wavg == 0.0) && (return 0.0)
  return (w1-w2)/wavg
end
_wdiff_rel(wmat1::Matrix{R},wmat2::Matrix{R}) where R<:Real = _wdiff_rel.(wmat1,wmat2)

function _plus_minus_rescale(mat::Array{R}) where R
  down,up = extrema(mat)
  down=abs(down)
  ret = copy(mat)
  for (i,x) in enumerate(ret)
    if x > 0
        ret[i] = x/up 
    elseif x < 0
        ret[i] = x/down
    end
  end
  return ret
end
function _wdiff_pm(wmat1::Matrix{R},wmat2::Matrix{R}) where R
  return _plus_minus_rescale(wmat1.-wmat2)
end

# +1 if both elements exist , 0 if both absent, -1 if mismatch
function _wdiff_structure(wmat1::Matrix{R},wmat2::Matrix{R}) where R
  ret = Matrix{Int64}(undef,size(wmat1)...)
  for i in eachindex(ret)
    if (wmat1[i] != 0.0) && (wmat2[i] != 0.0)
      ret[i] = 1
    elseif (wmat1[i] != 0.0) || (wmat2[i] != 0.0)
      ret[i] = -1
    else
      ret[i] = 0
    end
  end
  return ret
end

function compare_weigth_matrices(weights1::AbstractMatrix,
    weights2::AbstractMatrix,compare_funs...)
  if isempty(compare_funs)
    compare_funs=(_wdiff_plain,_wdiff_rel,_wdiff_pm)
  end
  w1 = Matrix(weights1)  
  w2 = Matrix(weights2)
  @assert size(w1) == size(w2)
  ret = map(compare_funs) do _f
    _f(w1,w2)
  end
  return ret
end



