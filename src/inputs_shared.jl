
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

# conductance based Poisson firing
struct NTPoissonCO <: NeuronType
  rate::Ref{Float64} # rate is in Hz and is mutable
  τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
  v_reversal::Float64 # reversal potential that affects postsynaptic neurons
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
function reset_input!(ps::PSPoisson)
  return nothing
end
function reset!(ps::PSPoisson)
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
  ratefunction::Function # sigature ::Float64 -> ::Vector{Float64}
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
  return PSInputPoissonFtMtuli(InputPoissonFtMulti(ratefun,τ),n,isfiring,isfi_alloc)
end
function reset!(ps::PSInputPoissonFtMulti)
  fill!(ps.isfiring,false)
  return nothing
end
function reset_input!(ps::PSInputPoissonFtMulti)
  return nothing
end
function local_update!(t_now::Float64,dt::Float64,ps::PSInputPoissonFtMulti)
  reset_spikes!(ps)
  Random.rand!(ps.isfiring_alloc)
  _rates::Vector{Float64} = ps.neurontype.ratefunction(t_now)
  @.  ps.isfiring = ps.isfiring_alloc < _rates*dt
  return nothing
end



## training with patterns with no consecutive repetitions
function _patterns_train_uniform(Npatt::Integer,Δt::Real,Ttot::Real)
  times = collect(0.0:Δt:Ttot)
  nt = length(times)
  nreps = ceil(Integer,nt/Npatt)
  last_patt = Npatt+1
  _seq = Vector{Vector{Int64}}(undef,nreps)
  for s in eachindex(_seq)
    _ss = shuffle(1:Npatt)
    while _ss[1] == last_patt
      shuffle!(_ss)
    end
    last_patt = _ss[end]
    _seq[s] = _ss
  end
  patt_seq = vcat(_seq...)[1:nt]
  return times,patt_seq
end

function binary_patterns_mat(Nneus::Integer,pattern_idx,low_val::Float64,
    high_val::Real)
  Npatt=length(pattern_idx)
  ret = fill(low_val,Nneus,Npatt+1)
  for p in 1:Npatt
    ret[pattern_idx[p].neupost,p] .= high_val
  end
  return ret
end