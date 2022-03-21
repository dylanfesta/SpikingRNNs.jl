```@meta
EditURL = "https://github.com/dylanfesta/SpikingRNNs.jl/blob/master/examples/lif_2neurons.jl"
```

# Two LIF neurons

In this example I show two LIF neuron, one excitatory and one inhibitory,
connected together. I plot the voltage traces, the internal currents, the
refractoriness.

I access to the internal variables directly, but in the next examples I will be using
recorder objects.

# Initialization

````@example lif_2neurons
using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:default)
using SparseArrays
using SpikingRNNs; const global S = SpikingRNNs

function onesparsemat(w::Real)
  return sparse(cat(w;dims=2))
end;
nothing #hide
````

# Parameters

````@example lif_2neurons
const dt = 1E-3
````

two LIF neurons, E and I

````@example lif_2neurons
const τe = 0.2 # time constant for dynamics
const τi = 0.1
const cap_e = τe # capacitance
const cap_i = τi
const vth = 10.  # action-potential threshold
const vreset = -5.0 # reset potential
const vleak = -5.0 # leak potential
const τrefre = 0.2 # refractoriness
const τrefri = 0.3
const τpcd = 0.2 # synaptic kernel decay

const ps_e = S.PSIFNeuron(1,τe,cap_e,vth,vreset,vleak,τrefre)
const ps_i = S.PSIFNeuron(1,τi,cap_i,vth,vreset,vleak,τrefri);
nothing #hide
````

## Define static inputs

````@example lif_2neurons
const h_in_e = 10.1 - vleak
const h_in_i = 0.0
const in_e = S.IFInputCurrentConstant([h_in_e,])
const in_i = S.IFInputCurrentConstant([h_in_i,])
````

## Define connections

connect E <-> I , both ways, but no autapses

````@example lif_2neurons
const conn_in_e = S.ConnectionIFInput([1.,])
const conn_in_i = S.ConnectionIFInput([1.,])
const w_ie = 30.0
const w_ei = 40.0
const conn_ie = S.ConnectionIF(τpcd,onesparsemat(w_ie))
const conn_ei = S.ConnectionIF(τpcd,onesparsemat(w_ei);is_excitatory=false);
nothing #hide
````

connected populations

````@example lif_2neurons
const pop_e = S.Population(ps_e,(conn_ei,ps_i),(conn_in_e,in_e))
const pop_i = S.Population(ps_i,(conn_ie,ps_e),(conn_in_i,in_i));
nothing #hide
````

that's it, let's make the network

````@example lif_2neurons
const network = S.RecurrentNetwork(dt,(pop_e,pop_i));

# # src
````

# Network simulation

````@example lif_2neurons
const Ttot = 15.
const times = (0:network.dt:Ttot)
nt = length(times)
````

set initial conditions

````@example lif_2neurons
ps_e.state_now[1] = vreset
ps_i.state_now[1] = vreset + 0.95*(vth-vreset)
````

things to save

````@example lif_2neurons
myvse = Vector{Float64}(undef,nt) # voltage
myfiringe = BitVector(undef,nt) # spike raster
myrefre = similar(myfiringe)  # if it is refractory
eicurr = similar(myvse)  # e-i current

myvsi = Vector{Float64}(undef,nt)
myfiringi = BitVector(undef,nt)
myrefri = similar(myfiringe)
iecurr = similar(myvsi)

for (k,t) in enumerate(times)
  S.dynamics_step!(t,network)
  myvse[k] = ps_e.state_now[1]
  myfiringe[k]=ps_e.isfiring[1]
  myrefre[k]=ps_e.isrefractory[1]
  myvsi[k] = ps_i.state_now[1]
  myfiringi[k]=ps_i.isfiring[1]
  myrefri[k]=ps_i.isrefractory[1]
  eicurr[k]=conn_ei.synaptic_kernel.trace[1]
  iecurr[k]=conn_ie.synaptic_kernel.trace[1]
end;
nothing #hide
````

add spikes for plotting purposes, the eight is set arbitrarily to
three times the firing threshold

````@example lif_2neurons
myvse[myfiringe] .= 3 * vth
myvsi[myfiringi] .= 3 * vth

theplot = let  plt=plot(times,myvse;leg=false,linewidth=1,
  ylabel="E (mV)",
  color=colorant"Midnight Blue") # the green line indicates when the neuron is refractory
  plot!(plt,times, 20.0*myrefre; opacity=0.6, color=:green,linewidth=1)
  plti=plot(times,myvsi;leg=false,linewidth=1,
     ylabel="I (mV)",
    color=colorant"Brick Red")
  plot!(plti,times, 20.0*myrefri; opacity=0.6, color=:green)
  pltcurr=plot(times, [ iecurr eicurr]; leg=false,
    linewidth=1, ylabel="E/I connection curr.",
    color=[colorant"Teal" colorant"Orange"])
  plot(plt,plti,pltcurr; layout=(3,1),
    xlabel=["" "" "time (s)"])
end;
plot(theplot)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

