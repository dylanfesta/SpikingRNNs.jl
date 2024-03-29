```@meta
EditURL = "https://github.com/dylanfesta/SpikingRNNs.jl/blob/master/examples/plasticity_triplet_2poisson.jl"
```

# Plasticity test for forced spikes

Plasticity between 2 "virtual" neurons The neurons are input neurons,
 i.e. their spikes are entirely driven and specified externally

````@example plasticity_triplet_2poisson
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default)
using SparseArrays
using SpikingRNNs; const global S = SpikingRNNs
function oneDSparse(w::Real)
  return sparse(cat(w;dims=2))
end

const dt = 1E-3
const Ttot = 10.0
````

start simple: two Poisson neurons
1 is connected to 2 and has some plasticity

````@example plasticity_triplet_2poisson
const rat1,rat2 = 2.,20.
const ps1 = S.PSPoisson(rat1,1E4,1)
const ps2 = S.PSPoisson(rat2,-1E4,1)
````

## Define a triplet-based plasticity rule

````@example plasticity_triplet_2poisson
const myplasticity = let scal=5E-2,
  τplus = 0.2,
  τminus = 0.2
  τx = 0.3
  τy = 0.3
  A2plus = scal*0.
  A3plus = scal*1.
  A2minus = -scal*5.
  A3minus = -scal*2.
  (n_post,n_pre) = (1,1)
  S.PlasticityTriplets(τplus,τminus,τx,τy,A2plus,A3plus,
    A2minus,A3minus,n_post,n_pre)
end
````

## Define the connection, populations , network
neuron 1 receives no connections
neuron 2 is connected to 1

````@example plasticity_triplet_2poisson
const wstart = 10.
const conn_2_1 = S.ConnectionPlasticityTest(oneDSparse(wstart),myplasticity)

const pop1 = S.UnconnectedPopulation(ps1)
const pop2 = S.Population(ps2,(conn_2_1,ps1))
const myntw = S.RecurrentNetwork(dt,pop1,pop2);
nothing #hide
````

### Set recorders: spikes and weights

````@example plasticity_triplet_2poisson
const Δt_wrec = Ttot/200
const rec_spikes1 = S.RecSpikes(ps1,50.0,Ttot)
const rec_spikes2 =  S.RecSpikes(ps2,50.0,Ttot)
const rec_weights = S.RecWeightsFull(conn_2_1,Δt_wrec,Ttot);
nothing #hide
````

## Run the network

````@example plasticity_triplet_2poisson
const times = (0:dt:Ttot)
const nt = length(times)
````

clean up, reset weights

````@example plasticity_triplet_2poisson
S.reset!.([rec_spikes1,rec_spikes2,rec_weights])
S.reset!.([ps1,ps2])
S.reset!.(conn_2_1.plasticities)

conn_2_1.weights[1,1] = wstart

for (k,t) in enumerate(times)
  rec_spikes1(t,k,myntw)
  rec_spikes2(t,k,myntw)
  rec_weights(t,k,myntw)
  S.dynamics_step!(t,myntw)
end
````

### Plot the spike raster alone

````@example plasticity_triplet_2poisson
theplot = let plt=plot(;leg=false,markersize=10),
  tra1 = S.get_spiketrains(rec_spikes1)
  tra2 = S.get_spiketrains(rec_spikes2)
  scatter!(plt,_->1,tra1,marker=:vline,color=:black,ylims=(0,3),
    markersize=20)
  scatter!(plt,_->2,tra2,marker=:vline,color=:black,ylims=(0,3),
    markersize=20 , xlabel="time (s)")
end;
plot(theplot)
````

### Plot weight change and spike raster

````@example plasticity_triplet_2poisson
theplot = let _recw = S.get_content(rec_weights)
  x=_recw.times
  y = [first(w) for w in _recw.weights_now]
  plt=plot(leg=false)
  plot!(plt,x,y;linewidth=2,leg=false,
    xlabel="time (s)")
  tra1 = S.get_spiketrains(rec_spikes1)
  tra2 = S.get_spiketrains(rec_spikes2)
  scatter!(twinx(plt),_->1,tra1 ;
     marker=:vline,color=:black,ylims=(0.5,3),leg=false)
  scatter!(twinx(plt),_->2,tra2 ;
     marker=:vline,color=:black,ylims=(0.5,3),leg=false,
      ylabel="synaptic weight")
end;
plot(theplot)
````

well, it looks like something!

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

