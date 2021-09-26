#=
Plasticity between 2 "virtual" neurons
The neurons are input neurons, i.e. their spikes are
entirely driven and specified externally
=#
push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
function oneDSparse(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

##
dt = 1E-3
Ttot = 10.0 
# start simple: two Poisson neurons
# 1 is connected to 2 and has some plasticity
ps1 = S.PSPoisson(12.,1E4,1)
ps2 = S.PSPoisson(20.,-1E4,1)


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

conn_2_1 = let w = oneDSparse(10.)
  S.ConnectionPlasticityTest(w,myplasticity)
end

# neuron 1 receives no connections
pop1 = S.UnconnectedPopulation(ps1)
# neuron 2 is connected to 1
pop2 = S.Population(ps2,(conn_2_1,ps1))

# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop1,pop2)

## recorders : spikes 
krec=1
rec_spikes1 = S.RecSpikes(ps1,50.0,Ttot)
rec_spikes2 =  S.RecSpikes(ps2,50.0,Ttot)
# all weights (but there is only one)
rec_weights = S.RecWeights(conn_2_1,krec,dt,Ttot)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_spikes1,rec_spikes2,rec_weights])
S.reset!.([ps1,ps2])
S.reset!.(conn_2_1.plasticities)
# initial conditions

for (k,t) in enumerate(times)
  rec_spikes1(t,k,myntw)
  rec_spikes2(t,k,myntw)
  rec_weights(t,k,myntw)
  S.dynamics_step!(t,myntw)
end

##

_ = let plt=plot(leg=false)
  scatter!(plt,_->1,rec_spikes1.spiketimes,marker=:vline,color=:white,ylims=(0.5,3))
  scatter!(plt,_->2,rec_spikes2.spiketimes,marker=:vline,color=:white,ylims=(0.5,3))
end

##

_ = let x=rec_weights.times
 y = rec_weights.weights_now[:]
 spkdict = S.get_spiketimes_dictionary(rec_spikes1)
 plt=plot(leg=false)
 plot!(plt,x,y;linewidth=2,leg=false)
 for (neu,spkt) in pairs(spkdict)
  scatter!(twinx(plt),_->neu,spkt ; 
    marker=:vline,color=:green,ylims=(0.5,3))
 end
 plt
end
