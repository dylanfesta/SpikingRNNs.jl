#=
Two LIF E neurons with a uni-directional plastic connection.
Forced to spike at precise times.
=#
push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools

##

dt = 1E-3
Ttot = 100.0
# LIF neuron
myτ = 0.2
vth = 10.
v_r = -5.0
τrefr= 0.0 # refractoriness
τpcd = 0.2 # post synaptic current decay
myinput = -5.0 # keep it low, only spike when forced to
ps_e = S.PSLIF(myτ,vth,v_r,τrefr,τpcd,2)

# one static input 
in_state_e = S.PSSimpleInput(S.InputSimpleOffset(myinput))
# connection will be FakeConnection()

# let's produce a couple of trains
train_epre = let rat = 1.0
  sort(rand(Uniform(0.05,Ttot),round(Integer,rat*Ttot) ))
end
train_epost = let rat = 0.5
  sort(rand(Uniform(0.05,Ttot),round(Integer,rat*Ttot) ))
end
# input population
ps_train_in=S.PSFixedSpiketrain([train_epre,train_epost],0.0)

# connection matrix is diagonal, train_epre to neuron pre
# train_epost to neuron post. Numbers just mark the presence of a directed edge
conn_e_in = let w_intrain2e = sparse([123. 0 ; 0 -456. ])
  S.ConnSpikeTransfer(w_intrain2e)
end

# now the *plastic* connection between 2 neurons
myplasticity = let τplus = 0.2,
  τminus = 0.2
  τx = 0.3
  τy = 0.3
  A2plus = 0.01
  A3plus = 0.01
  A2minus = 0.02
  A3minus = 0.02
  (n_post,n_pre) = (2,2) 
  S.PlasticityTriplets(τplus,τminus,τx,τy,A2plus,A3plus,
    A2minus,A3minus,n_post,n_pre)
end

conn_e2e = let w_e2e = sparse([0.0 0.0 ; 0.5 0.0 ])
  S.ConnGeneralIF(w_e2e,myplasticity)
end

##
# the input evolves in time, so it's a population too
pop_in = S.UnconnectedPopulation(ps_train_in)
# then the E population is connected to the input trains and to itself
pop_e = S.Population(ps_e,(conn_e_in,ps_train_in),(conn_e2e,ps_e))

# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop_in,pop_e)

## recorders : spikes 
rec_spikes_e = S.RecSpikes(ps_e,50.0,Ttot)
rec_spikes_in =  S.RecSpikes(ps_train_in,50.0,Ttot)
# internal potential
krec = 1
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)
# all weights (but there is only one)
rec_weights = S.RecWeights(conn_e2e,krec,dt,Ttot)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e,rec_spikes_in,rec_weights])
S.reset!.([ps_e,ps_train_in])
S.reset!.(conn_e2e.plasticities)
# initial conditions
ps_e.state_now[1] = 0.0

for (k,t) in enumerate(times)
  rec_state_e(t,k,myntw)
  rec_spikes_e(t,k,myntw)
  rec_spikes_in(t,k,myntw)
  rec_weights(t,k,myntw)
  S.dynamics_step!(t,myntw)
end
# this is useful for visualization only
S.add_fake_spikes!(1.5vth,rec_state_e,rec_spikes_e)
##
_ = let x=rec_weights.times
 y = rec_weights.weights_now[:]
 spkdict = S.get_spiketimes_dictionary(rec_spikes_e)
 plt=plot(leg=false)
 plot!(plt,x,y;linewidth=2,leg=false)
 for (neu,spkt) in pairs(spkdict)
  scatter!(twinx(plt),_->neu,spkt ; 
    marker=:vline,color=:green,ylims=(0.5,3))
 end
 plt
end

