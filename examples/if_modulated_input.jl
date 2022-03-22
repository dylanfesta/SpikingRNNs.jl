#=
# LIF neurons receive a sinusoidal spiking input

The neurons are independent from each other, they also receive independent 
input trains, modulated sinusoidally.
=#
push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using FileIO

## # src

# Time
const dt = 1E-3
const Ttot = 5.0;

# ## Create 10  excitatory LIF neurons

const N = 10
const τ = 0.2 # time constant for dynamics 
const cap = τ # capacitance
const vth = 10.  # action-potential threshold 
const vreset = -5.0 # reset potential
const vleak = -5.0 # leak potential
const τrefr = 0.0 # refractoriness
const τpcd = 0.02 # synaptic kernel decay

const ps = S.PSIFNeuron(N,τ,cap,vth,vreset,vleak,τrefr);

# ## Modulation signal
const ω = 1.0
const text_start = 0.23 # when the signal is on
const rext_min,rext_max = 10,200 
const rext_off = 6.0

function ratefun(t::Float64)
  (t<=text_start) && (return rext_off)
  return rext_min + (0.5+0.5sin(2π*t/ω))*(rext_max-rext_min)
end
function ratefun_upper(::Float64)
  return rext_max # slightly suboptimal when it comes to generation of spikes 
end

# ## Input object
const ps_input = S.IFInputSpikesFunScalar(N,ratefun,ratefun_upper);

#=
connection from input to E : the synaptic kernel is conductance based.
=#
const τker = 0.3
const vrev_in = 15.0 # must be higher than firing threshold!
const in_ker = S.SyKConductanceExponential(N,τker,vrev_in)
const win = 0.1
const in_weights = fill(win,N)
const conn_e_in = S.ConnectionIFInput(in_weights,in_ker)

# Now I can define the population and the network. The neurons have no mutual connections, they are independent

const pop = S.Population(ps,(conn_e_in,ps_input))
const network = S.RecurrentNetwork(dt,pop)


## I will record the full spike train for the neurons.
const rec_spikes = S.RecSpikes(ps,50.0,Ttot)
# and the internal potential
const krec = 1
const rec_state = S.RecStateNow(ps,krec,dt,Ttot)

const times = (0:dt:Ttot)
const nt = length(times);

## #src
# ## Run the network


S.reset!.([rec_state,rec_spikes])
S.reset!(ps);
fill!(ps.state_now,0.0)

for (k,t) in enumerate(times)
  rec_state(t,k,network)
  rec_spikes(t,k,network)
  S.dynamics_step!(t,network)
end;
# this is useful for visualization only
S.add_fake_spikes!(1.5vth,rec_state,rec_spikes);

## #src
# ## Plot internal potential for a pair of neurons

_ = let neu1=1,neu2=2,
  times = rec_state.times,
  mpot1 = rec_state.state_now[neu1,:]
  mpot2 = rec_state.state_now[neu2,:]
  plot(times,[mpot1 mpot2], 
    xlabel="time (s)",
    ylabel="membrane potential (mV)",
    label=["neuron 1" "neuron 2"],
    leg=:bottomright)
end

## #src
# ## Plot train raster
const trains = S.get_spiketrains(rec_spikes)

theraster = let rdt = 0.01,
  rTend = Ttot
  S.draw_spike_raster(trains,rdt,rTend)
end

#=
The raster might not be visible online, or it might have low quality,
but if you run this script locally, you can save it as a png image as follows:
`save("<save path>",theraster)`
=#

save("/tmp/rast.png",theraster) #src

## Publish ! #src
using Literate #src
Literate.markdown(@__FILE__,"docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/SpikingRNNs.jl/blob/master") #src