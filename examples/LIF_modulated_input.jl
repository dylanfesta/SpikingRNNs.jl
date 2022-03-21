#=
Two LIF E neurons receive a sinusoidal type of input 
from am input population
=#
push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools

## # src

# Generate 10  excitatory LIF neurons
## #src
const N = 10
const dt = 1E-3
# two LIF neurons, E and I
const τ = 0.2 # time constant for dynamics 
const cap = τ # capacitance
const vth = 10.  # action-potential threshold 
const vreset = -5.0 # reset potential
const vleak = -5.0 # leak potential
const τrefr = 0.2 # refractoriness
const τpcd = 0.2 # synaptic kernel decay

const ps = S.PSIFNeuron(N,τe,cap_e,vth,vreset,vleak,τrefr)



# Modulation signal
const ω = 20.0
const text_start = 0.23 # when the signal is on
const rext_min,rext_max = 40,80 

function ratefun(t::Float64)
  (t<=text_start) && (return 0.0)
  return rext_min + (0.5+0.5sin(2π*t/ω))*(rext_max-rext_min)
end

##

# connection from input to E
# 5 inputs go to one neuron, 5 to the other
w_e_input = sparse(vcat([ones(5)... , zeros(5)...]',
    [zeros(5)... , ones(5)...]'))
lmul!(0.33,w_e_input)
conn_e_in = S.ConnGeneralIF(w_e_input)
## Populations
 # the input evolves in time, so it's a population too
pop_in = S.UnconnectedPopulation(ps_in)
pop_e = S.Population(ps_e,(conn_e_in,ps_in))

myntw = S.RecurrentNetwork(dt,pop_in,pop_e)

## recorders : spikes 
rec_spikes_e = S.RecSpikes(ps_e,50.0,Ttot)
rec_spikes_in =  S.RecSpikes(ps_in,50.0,Ttot)
# internal potential
krec = 1
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e,rec_spikes_in])
S.reset!.([ps_e,ps_in])
# initial conditions
fill!(ps_e.state_now,0.0)

for (k,t) in enumerate(times)
  rec_state_e(t,k,myntw)
  rec_spikes_e(t,k,myntw)
  rec_spikes_in(t,k,myntw)
  S.dynamics_step!(t,myntw)
end
# this is useful for visualization only
S.add_fake_spikes!(1.5vth,rec_state_e,rec_spikes_e)
##

spk_in_dict = S.get_spiketimes_dictionary(rec_spikes_in)

_ = let plt=plot(leg=false)
 for (neu,spkt) in pairs(spk_in_dict)
  scatter!(plt,_->neu,spkt ; 
    marker=:vline,color=:white,ylims=(0.5,10.5))
 end
 plt
end

##
S.get_mean_rates(rec_spikes_in,dt,Ttot)

## 
_ = let plt=plot(leg=false)
  x=rec_state_e.times
  y = rec_state_e.state_now
  plot!(x,[y[1,:] y[2,:]])
end
