
##############
# Inputs 



# Constant input current (mostly for testing purposes)
# the scaling is regulated here, and the weight is ignored
struct InputSimpleOffset <: NeuronType
  α::Float64 # scaling constant
end


# when the presynaptic is a simple input, just sum linearly to the input vector
function forward_signal!(tnow::Real,dt::Real,p_post::PopulationState,
    c::Connection,p_pre::PSSimpleInput{InputSimpleOffset})
  for i in eachindex(p_post.input)
    p_post.input[i] += p_pre.neurontype.α
  end
  return nothing
end

# if the receiver is a spiking type, update only when NOT refractory
function forward_signal!(tnow::Real,dt::Real,p_post::PSSpikingType,
    c::Connection,p_pre::PSSimpleInput{InputSimpleOffset})
  for i in eachindex(p_post.input)
    if ! p_post.isrefractory[i]
      p_post.input[i] += p_pre.neurontype.α
    end
  end
  return nothing
end


# independent Gaussian noise for each neuron
# weight is ignored
struct InputIndependentNormal <: NeuronType
  α::Float64 # scaling constant
end

# the connection is ignored: all neurons of the same population receive
# the same noise. Postsynaptic neurons must have a τ
function forward_signal!(tnow::Real,dt::Real,p_post::PopulationState,
    c::Connection,p_pre::PSSimpleInput{InputIndependentNormal})
  # std is α for isolated neuron
  _reg = sqrt(2*p_post.neurontype.τ/dt)
  for i in eachindex(p_post.input)
    p_post.input[i] += p_pre.neurontype.α*_reg*randn()
  end
  return nothing
end


# Arbitrary 1D time-dependent function
# injected directly as input current
# the input is regulated by the weight matrix

struct InputSomeFunction <: NeuronType
  f::Function # f(t) = input
end
function forward_signal!(tnow::Real,dt::Real,p_post::PopulationState,
    c::Connection,p_pre::PSSimpleInput{InputSomeFunction})
  in_t::Float64 = p_pre.neurontype.f(tnow)
  ws = nonzeros(c.weights)
  w_idx = rowvals(c.weights)
  for (i,w) in zip(w_idx,ws)
    p_post.input[i] += w*in_t
  end
  return nothing
end

# Fixed spiketrain as input
struct InputFixedSpiketrain <: NeuronType
  trains::Vector{Vector{Float64}} # one neuron -> one vector of spiketimes
  τ_output_decay::Float64
end
struct PSFixedSpiketrain{NT} <: PSSpikingType{NT where NT <: InputFixedSpiketrain}
  neurontype::NT
  n::Int64 # pop size
  isfiring::BitArray{1}
  counter::Vector{Int64} # keep track of elapsed spikes (instead of doing a full search)
end

function PSFixedSpiketrain(nt::InputFixedSpiketrain)
  n = length(nt.trains)
  isfiring=falses(n)
  counter=ones(n)
  return PSFixedSpiketrain{InputFixedSpiketrain}(nt,n,isfiring,counter)
end
function PSFixedSpiketrain(train::Vector{Vector{Float64}},τ::Real)
  return PSFixedSpiketrain(InputFixedSpiketrain(train,τ))
end
function reset_input!(ps::PSFixedSpiketrain)
  return nothing
end

function local_update!(t_now::Float64,dt::Float64,ps::PSFixedSpiketrain)
  reset_spikes!(ps)
  for neu in (1:ps.n) # WARNING : at most 1 spike per neuron in 1 dt
    # pick the spiketrain of each neuron, at the counter index
    c = ps.counter[neu]
    @inbounds if checkbounds(Bool,ps.neurontype.trains[neu],c)
      t_signal = ps.neurontype.trains[neu][c]
      if t_signal < t_now+dt
        ps.isfiring[neu] = true # neuron is firing
        ps.counter[neu] += 1    # move counter forward
      end
    end
  end
  return nothing
end
function reset!(ps::PSFixedSpiketrain)
  reset_spikes!(ps)
  fill!(ps.counter,1)
  return nothing
end


# Poisson firing
struct NTPoisson <: NeuronType
  rate::Ref{Float64} # rate is in Hz and is mutable
  τ_output_decay::Float64 # decay of postsynaptic conductance
end

struct PSPoisson{NT} <: PSSpikingType{NT}
  neurontype::NT
  n::Int64 # pop size
	isfiring::BitArray{1} # firing will be i.i.d. Poisson
  isfiring_alloc::Vector{Float64} # allocate probabilities
end
function PSPoisson(p::NT,n) where NT<:NeuronType
  return PSPoisson{NT}(p,n,falses(n),zeros(Float64,n))
end
function PSPoisson(rate::Real,τ_decay::Real,n::Integer)
  nt = NTPoisson(rate,τ_decay)
  return PSPoisson{NTPoisson}(nt,n,falses(n),zeros(Float64,n))
end
function reset_input!(::PSPoisson)
  return nothing
end
function reset!(::PSPoisson)
  return nothing
end
function local_update!(t_now::Float64,dt::Float64,ps::PSPoisson)
  reset_spikes!(ps)
  Random.rand!(ps.isfiring_alloc)
  _rate = ps.neurontype.rate[]
  @.  ps.isfiring = ps.isfiring_alloc < _rate*dt
  return nothing
end

# Poisson with frequency folloing some function

struct InputPoissonFt <: NeuronType
  ratefunction::Function # sigature ::Float64 -> Float64
  τ_output_decay::Float64
end
struct PSInputPoissonFt{NT} <: PSSpikingType{NT}
  neurontype::NT
  n::Int64 # pop size
	isfiring::BitArray{1} 
  isfiring_alloc::Vector{Float64} # allocate probabilities
end
function PSInputPoissonFt(ratefun,τ,n)
  isfiring = falses(n)
  isfi_alloc = zeros(n)
  return PSInputPoissonFt(InputPoissonFt(ratefun,τ),n,isfiring,isfi_alloc)
end

function reset!(ps::PSInputPoissonFt)
  fill!(ps.isfiring,false)
  return nothing
end
function reset_input!(ps::PSInputPoissonFt)
  return nothing
end

function local_update!(t_now::Float64,dt::Float64,ps::PSInputPoissonFt)
  reset_spikes!(ps)
  _rate = ps.neurontype.ratefunction(t_now)
  Random.rand!(ps.isfiring_alloc)
  @.  ps.isfiring = ps.isfiring_alloc < _rate*dt
  return nothing
end

# Poisson with frequency following a *multi dimensional* function
# useful for input patterns

struct InputPoissonFtMulti <: NeuronType
  ratefunction::Function # signature ::Float64 -> ::Vector{Float64}
  τ_output_decay::Float64
end
struct PSInputPoissonFtMulti{NT} <: PSSpikingType{NT}
  neurontype::NT
  n::Int64 # pop size
	isfiring::BitArray{1} 
  isfiring_alloc::Vector{Float64} # allocate probabilities
end
function PSInputPoissonFtMulti(ratefun,τ,n)
  isfiring = falses(n)
  isfi_alloc = zeros(n)
  return PSInputPoissonFtMulti(InputPoissonFtMulti(ratefun,τ),n,isfiring,isfi_alloc)
end
function reset!(ps::PSInputPoissonFtMulti)
  fill!(ps.isfiring,false)
  return nothing
end
function reset_input!(ps::PSInputPoissonFtMulti)
  return nothing
end
function local_update!(t_now::Float64,dt::Float64,ps::PSInputPoissonFtMulti)
  #reset_spikes!(ps)  not needed, I rewrite it fully
  Random.rand!(ps.isfiring_alloc)
  _rates::Vector{Float64} = ps.neurontype.ratefunction(t_now)
  @.  ps.isfiring = ps.isfiring_alloc < _rates*dt
  return nothing
end

# helper function for
# training with patterns with no consecutive repetitions
function _generate_uniform_pattern_sequence(Npatterns::Integer,Ntrials::Integer)
  nreps = ceil(Integer,Ntrials/Npatterns)
  last_patt = 0
  _seq = Vector{Int64}[]
  for _ in 1:nreps
    _ss = shuffle(1:Npatterns)
    while _ss[1] == last_patt
      shuffle!(_ss)
    end
    last_patt = _ss[end]
    push!(_seq,_ss)
  end
  patt_seq = vcat(_seq...)[1:Ntrials]
  return patt_seq
end

# Define separate primitive for low-level tuning
function _make_pattern_function(pattern_sequence::Vector{<:Integer},
    pattern_times::Vector{R},full_patterns::Matrix{R}) where R
  return function (t::R,i::Integer)
    it = searchsortedfirst(pattern_times,t)-1
    if checkbounds(Bool,pattern_sequence,it)
      idxpatt = pattern_sequence[it]
      return full_patterns[i,idxpatt]
    else
      return full_patterns[i,end]
    end
  end 
end

# but is it a functor ?
function pattern_functor(Δt::R,Ttot::R,
    low::R,high::R,
    npost::Integer,
    idxs_patternpop::Vector{Vector{Int64}} ; 
    Δt_pattern_blank::R=0.0,
    t_pattern_delay::R=0.0) where R<:Real
  # time vector
  ts_pattern_start = collect(range(t_pattern_delay,Ttot;step=Δt+Δt_pattern_blank))
  # make pattern sequence
  Npatt = length(idxs_patternpop)
  patt_seq = _generate_uniform_pattern_sequence(Npatt,length(ts_pattern_start)-1)
  # must add blanks
  if Δt_pattern_blank > 0.0
    blank_seq = fill(Npatt+1,length(patt_seq))
    patt_seq = [transpose(hcat(patt_seq,blank_seq))...]
    patttimes = repeat(ts_pattern_start;inner=2)[2:end]
    for i in 2:2:length(patttimes)
      patttimes[i] -=Δt_pattern_blank
    end
  else
    patttimes = ts_pattern_start
  end
  # Generate full scale patterns
  fullp = fill(low,(npost,Npatt+1))
  for (i,neuidxs) in enumerate(idxs_patternpop)
    fullp[neuidxs,i] .= high
  end
  # generate the function and return it
  return _make_pattern_function(patt_seq,patttimes,fullp)
end
function pattern_functor_upperlimit(low::R,high::R,
    Ntot::Integer,
    idxs_patternpop::Vector{Vector{Int64}}) where R
  theref = fill(low,Ntot)
  for idxs_patt in idxs_patternpop
    theref[idxs_patt] .= high
  end
  return function(::Real,i::Integer)
    return theref[i]
  end
end


## generalize a little on the Inputs

abstract type SpikeGenerator end

struct SGPoisson <: SpikeGenerator
  rate::Float64
end
function generate_spikes!(t_now::Float64,dt::Float64,isfiring::BitArray{1},sg::SGPoisson)
  _rat = sg.rate*dt
  @inbounds @simd for i in eachindex(isfiring)
    isfiring[i] = rand() < _rat
  end
  return nothing
end
struct SGPoissonF <: SpikeGenerator
  ratefun::Function # t::Float64 -> rate::Float64
end
function generate_spikes!(t_now::Float64,dt::Float64,isfiring::BitArray{1},sg::SGPoissonF)
  _rat::Float64 = sg.ratefun(t_now)*dt
  if _rat > 0.0
    @inbounds @simd for i in eachindex(isfiring)
      isfiring[i] = rand() < _rat
    end
  else
    fill!(false,isfiring)
  end
  return nothing
end
struct SGPoissonMulti <: SpikeGenerator
  rates::Vector{Float64}
end
function generate_spikes!(t_now::Float64,dt::Float64,isfiring::BitArray{1},sg::SGPoissonMulti)
  @inbounds @simd for i in eachindex(isfiring)
    isfiring[i] = rand() < sg.rates[i]*dt
  end
  return nothing
end
struct SGPoissonMultiF <: SpikeGenerator
  ratefun::Function # t::Float64 -> rates::Vector{Float64}
end
function generate_spikes!(t_now::Float64,dt::Float64,isfiring::BitArray{1},sg::SGPoissonMultiF)
  _rates::Vector{Float64} = sg.ratefun(t_now) 
  @inbounds @simd for i in eachindex(isfiring)
    if _rates[i] > 0
      isfiring[i] = rand() < _rates[i]*dt
    else
      isfiring[i] = false
    end
  end
  return nothing
end

# for exact spike generation, I need an upper bound for the instantaneous rate
struct SGPoissonFExact <: SpikeGenerator
  ratefunction::Function # (t::Float64,i::Int64) -> rate_i::Float64
  ratefunction_upper::Function # (t::Float64,i::Int64) -> rate_i::Float64
end

struct SGTrains <: SpikeGenerator
  trains::Vector{Vector{Float64}}
  counter::Vector{Int64}
  function SGTrains(trains)
    counter = fill(1,length(trains))
    new(trains,counter)
  end
end
function generate_spikes!(t_now::Float64,dt::Float64,isfiring::BitArray{1},sg::SGTrains)
  fill!(isfiring,false)
  for i in eachindex(isfiring)
    # pick the spiketrain of each neuron, at the counter index
    # I could have used searchsortedfirst, too
    c = sg.counter[i]
    if checkbounds(Bool,sg.trains[i],c) 
      @inbounds t_next_spike = sg.trains[i][c]
      if t_next_spike < t_now+dt
        isfiring[i] = true # neuron is firing
        sg.counter[i] += 1    # move counter forward
      end
    end
  end
  return nothing
end

struct NTInputConductance{SK<:SynapticKernel,SG<:SpikeGenerator} <:NTConductance
  spikegenerator::SG
  synaptic_kernel::SK
  v_reversal::Float64 # reversal potential that affects postsynaptic neurons
end
struct PSInputConductance{NT<:NTInputConductance} <: PSSpikingType{NT}
  n::Int64
  neurontype::NT
  isfiring::BitArray{1}
  function PSInputConductance(nt::NT,n_neu::Int64) where NT<:NeuronType
    isfiring=falses(n_neu)
    new{NT}(n_neu,nt,isfiring)
  end
end
reset_input!(::PSInputConductance) = nothing
function reset!(::PSInputConductance{NT}) where {SK,SG,NT<:NTInputConductance{SK,SG}}
  return nothing
end
function reset!(ps::PSInputConductance{NT}) where {SK,NT<:NTInputConductance{SK,SGTrains}}
  fill!(ps.neurontype.spikegenerator.counter,1)
  return nothing
end
function local_update!(t_now::Float64,dt::Float64,ps::PSInputConductance)
  generate_spikes!(t_now,dt,ps.isfiring,ps.neurontype.spikegenerator)
  return nothing
end

##### 
# here I also supplement the Forward signal , for simplicity and efficiency
# works with FakeConnection only
# downside : one needs to include the population weight here (vector of weights)
# other downside : NO PLASTICITY
# note that input could be correlated, if I define the spike generation in 
# NTInputConductance accordingly ... but it becomes hacky 

struct PSInputPoissonConductance{NT<:NTInputConductance} <: PSSpikingType{NT}
  n::Int64 # Must be the same as npost (n neurons that receive the input)
  neurontype::NT
  isfiring::BitArray{1}
  input_weights::Vector{Float64} 
  trace1::Vector{Float64}  # alas! traces are still needed
  trace2::Vector{Float64}
  function PSInputPoissonConductance(nt::NT,weights::Vector{Float64}) where NT<:NeuronType
    n_neu = length(weights)
    isfiring=falses(n_neu)
    tr1 = zeros(n_neu)
    tr2 = zeros(n_neu)
    new{NT}(n_neu,nt,isfiring,weights,tr1,tr2)
  end
  function PSInputPoissonConductance(nt::NT,weight::Float64,n_neu::Integer) where NT<:NeuronType
    weights = fill(weight,n_neu)
    isfiring=falses(n_neu)
    tr1 = zeros(n_neu)
    tr2 = zeros(n_neu)
    new{NT}(n_neu,nt,isfiring,weights,tr1,tr2)
  end
end
function reset!(ps::PSInputPoissonConductance{NT}) where {SK,SG,NT<:NTInputConductance{SK,SG}}
  fill!(ps.trace1,0.0)
  fill!(ps.trace2,0.0)
  fill!(ps.isfiring,false)
  return nothing
end
# special treatment for train inputs
function reset!(ps::PSInputPoissonConductance{NT}) where {SK,NT<:NTInputConductance{SK,SGTrains}}
  fill!(ps.trace1,0.0)
  fill!(ps.trace2,0.0)
  fill!(ps.isfiring,false)
  fill!(ps.neurontype.spikegenerator.counter,1)
  return nothing
end

@inline function trace_decay!(dt::Real,ps::PSInputPoissonConductance)
  trace_decay!(dt,ps.trace1,ps.trace2,ps.neurontype.synaptic_kernel)
end

function forward_signal!(t_now::Real,dt::Real,
      pspost::PSLIFConductance,::FakeConnection,
      pspre::PSInputPoissonConductance)
  preneu = pspre.neurontype
  pre_v_reversal = preneu.v_reversal
  pre_synker = preneu.synaptic_kernel
  # traces time decay (here or at the end? meh)
  trace_decay!(dt,pspre)
  # update spikes 
  generate_spikes!(t_now,dt,pspre.isfiring,preneu.spikegenerator)
  @inbounds for (i,firing) in enumerate(pspre.isfiring)
    # update trace if firing
    if firing
      # increment traces of spiking neurons
      # function defined in lif_conductance.jl
      trace_spike_update!(pspre.input_weights[i],pspre.trace1,pspre.trace2,pre_synker,i)
    end
    # update ALL inputs with synaptic traces, unless refractory
    if !pspost.isrefractory[i]
      # function defined in lif_conductance.jl
      postsynaptic_kernel_update!(pspost.input,pspost.state_now,
          pspre.trace1,pspre.trace2,pre_synker,pre_v_reversal,i)
    end
  end
  return nothing
end

#####
# but what happens when the input rate is >> dt ? 
# reducing dt would be extremely unefficient... so
# NEW input implementation wiht EXACT spike timing
# and multiple update of traces when needed


# Thinning algorith, e.g.  Laub,Taimre,Pollet 2015
function _rand_by_thinning(t_start::Real,get_rate::Function,get_rate_upper::Function;
    Tmax=50.0,nowarning::Bool=false)
  t = t_start 
  while (t-t_start)<Tmax # Tmax is upper limit, if rate too low 
    rup = get_rate_upper(t)
    Δt = rand(Exponential())./rup
    t = t+Δt
    u = rand(Uniform(0.0,rup))
    if u <= get_rate(t) 
      return t
    end
  end
  # if we are above Tmax, just return upper limit
  if !nowarning
    @warn "Upper limit reached, input firing below $(inv(Tmax)) Hz"
  end
  return Tmax + t_start
end

struct PSInputPoissonConductanceExact{NT<:NTInputConductance} <: PSSpikingType{NT}
  n::Int64 # Must be the same as npost (n neurons that receive the input)
  neurontype::NT
  firingtimes::Vector{Float64}
  input_weights::Vector{Float64} 
  trace1::Vector{Float64}  # alas! traces are still needed
  trace2::Vector{Float64}
  function PSInputPoissonConductanceExact(nt::NT,weights::Vector{Float64}) where NT<:NeuronType
    n_neu = length(weights)
    tr1,tr2,firingtimes = ntuple(_->Vector{Float64}(undef,n_neu),3)
    ret = new{NT}(n_neu,nt,firingtimes,weights,tr1,tr2)
    reset!(ret)
    return ret
  end
  function PSInputPoissonConductanceExact(nt::NT,weight::Float64,n_neu::Integer) where NT<:NeuronType
    weights = fill(weight,n_neu)
    tr1,tr2,firingtimes = ntuple(_->Vector{Float64}(undef,n_neu),3)
    ret = new{NT}(n_neu,nt,firingtimes,weights,tr1,tr2)
    reset!(ret)
    return ret
  end
end

function _get_spiketime_update(t_current_spike::Real,sg::SGPoisson,::Integer)
  return t_current_spike + rand(Exponential())/sg.rate
end

function _get_spiketime_update(t_current_spike::Real,sg::SGPoissonMulti,i::Integer)
  return t_current_spike + rand(Exponential())/sg.rates[i]
end

function _get_spiketime_update(::Real,sg::SGTrains,i::Integer)
  # note that t_current_spike is expected to be 
  # sg.trains[i][counter[i]] (before counter update)
  c = sg.counter[i]
  if checkbounds(Bool,sg.trains[i],c+1) 
    sg.counter[i] = c+1    # move counter forward
    return sg.trains[i][c+1] # return next spike time
  else
    return Inf
  end 
end
function _get_spiketime_update(::Real,sg::SGPoissonMultiF,i::Integer)
  return error("please define input using `SGPoissonFExact`")
end

function _get_spiketime_update(t_current_spike::Real,sg::SGPoissonFExact,i::Integer)
  return _rand_by_thinning(t_current_spike,
    t->sg.ratefunction(t,i),t->sg.ratefunction_upper(t,i))
end

# in this case reset is an initialization step... 
# which means I need to compute all first spike proposals


function reset!(ps::PSInputPoissonConductanceExact{NT}) where {SK,SG,NT<:NTInputConductance{SK,SG}}
  fill!(ps.trace1,0.0)
  fill!(ps.trace2,0.0)
  sgen = ps.neurontype.spikegenerator
  for i in eachindex(ps.firingtimes)
    ps.firingtimes[i] = _get_spiketime_update(0.0,sgen,i)
  end
  # special treatment for train inputs
  if hasproperty(sgen,:counter)
    fill!(sgen.counter,1)
  end
  return nothing
end

@inline function trace_decay!(dt::Real,ps::PSInputPoissonConductanceExact)
  trace_decay!(dt,ps.trace1,ps.trace2,ps.neurontype.synaptic_kernel)
end

function forward_signal!(t_now::Real,dt::Real,
      pspost::PSLIFConductance,::FakeConnection,
      pspre::PSInputPoissonConductanceExact)
  preneu = pspre.neurontype
  pre_v_reversal = preneu.v_reversal
  pre_synker = preneu.synaptic_kernel
  sgen = pspre.neurontype.spikegenerator
  # traces time decay (here or at the end? meh)
  trace_decay!(dt,pspre)
  # this is an alternative to the code below 
  # # if t_now moved past a spiketime...
  # idxs_past = findall(t_now .>= pspre.spiketimes)
  # while !isempty(idxs_past)
  #   for i in idxs_past
  #     # increment traces of spiking neurons
  #     # function defined in lif_conductance.jl
  #     trace_spike_update!(pspre.input_weights[i],pspre.trace1,pspre.trace2,pre_synker,i)
  #     # update spiketime
  #     pspre.spiketimes[i] = _get_spiketime_update(t_spike,sgen,i)
  #   end
  #   idxs_past = findall(t_now .>= pspre.spiketimes)
  # end
  for i in eachindex(pspre.firingtimes)
    tspike = pspre.firingtimes[i]
    weight_i =  pspre.input_weights[i]
    while t_now >= tspike
      # increment traces of spiking neurons
      # function defined in lif_conductance.jl
      trace_spike_update!(weight_i,pspre.trace1,pspre.trace2,pre_synker,i)
      # update spiketime
      tspike = _get_spiketime_update(tspike,sgen,i)
    end
    pspre.firingtimes[i]=tspike
  end
  @inbounds @simd for i in eachindex(pspost.input)
    if ! pspost.isrefractory[i]
    # function defined in lif_conductance.jl
    postsynaptic_kernel_update!(pspost.input,pspost.state_now,
            pspre.trace1,pspre.trace2,pre_synker,pre_v_reversal,i)
    end
  end
  return nothing
end


#################################

# Inhibitory stabilization

# from Fiete et al , 2010 , Neuron
# not a firing neuron type! More like an input type 
# all constants here, works with FakeConnection only
struct NTConductanceInputInhibitoryStabilization <:NeuronType
  v_reversal::Float64
  Aglo::Float64
  Aloc::Float64
  τglo::Float64
  τloc::Float64
end
struct PSConductanceInputInhibitionStabilization <: PopulationState{NTConductanceInputInhibitoryStabilization}
  neurontype::NTConductanceInputInhibitoryStabilization
  n::Int64 # size of stabilized E population
  post_loc_traces::Vector{Float64}
  post_glo_trace::Ref{Float64}
end
function PSConductanceInputInhibitionStabilization(
    v_reversal::R,Aglo::R,Aloc::R,τglo::R,τloc::R,n::Integer) where R
  nt = NTConductanceInputInhibitoryStabilization(v_reversal,Aglo,Aloc,τglo,τloc)
  loc_traces = zeros(n)
  glo_trace = Ref(0.0)
  return PSConductanceInputInhibitionStabilization(nt,n,
    loc_traces,glo_trace)
end

function reset!(ps::PSConductanceInputInhibitionStabilization)
  fill!(ps.post_loc_traces,0.0)
  ps.post_glo_trace[]=0.0
  return nothing
end

# traces decay in time
@inline function trace_decay!(dt::Real,ps::PSConductanceInputInhibitionStabilization)
  ps.post_loc_traces .*= exp(-dt/ps.neurontype.τloc)
  ps.post_glo_trace[] *= exp(-dt/ps.neurontype.τglo) 
end

# adds 1 to traces
function trace_spike_update!(ps::PSConductanceInputInhibitionStabilization,
    isfiring::BitArray{1})
  @inbounds for (i,firing) in enumerate(isfiring)
    if firing
      ps.post_loc_traces[i] += 1.0
      ps.post_glo_trace[] += 1.0
    end
  end
  return nothing
end

# This function should never be called anyway!
# because this PS does not need to appear as pre in a Population object 
function local_update!(::Float64,::Float64,
      ::PSConductanceInputInhibitionStabilization)
  return nothing # nothing to update, traces decay in the forward signal function
end

function forward_signal!(t_now::Real,dt::Real,
      pspost::PSLIFConductance,::FakeConnection,
      pspre::PSConductanceInputInhibitionStabilization)
  preneu = pspre.neurontype
	# add the currents to postsynaptic input
	# ONLY non-refractory neurons
	@inbounds @simd for i in eachindex(pspost.input)
    if ! pspost.isrefractory[i]
      ker_loc = preneu.Aloc*pspre.post_loc_traces[i]/preneu.τloc
      ker_glo = preneu.Aglo*pspre.post_glo_trace[]/(pspre.n*preneu.τglo)
	    pspost.input[i] += (ker_loc+ker_glo)*(preneu.v_reversal - pspost.state_now[i])
		end
	end
  # traces time decay
  trace_decay!(dt,pspre)
  # update traces with spiking neurons
	trace_spike_update!(pspre,pspost.isfiring)
  # all done !
  return nothing
end