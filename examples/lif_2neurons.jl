using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs

function onesparsemat(w::Real)
  return sparse(cat(w;dims=2))
end

##
const dt = 1E-3
# two LIF neurons, E and I
const τe = 0.2 # time constant for dynamics 
const τi = 0.1
const cap = 1.0 # capacitance
const vth = 10.  # action-potential threshold 
const vreset = -5.0 # reset potential
const vleak = -5.0 # leak potential
const τrefre = 0.3 # refractoriness
const τrefri = 0.1 
const τpcd = 0.2 # synaptic kernel decay

const ps_e = S.PSIFNeuron(1,τe,cap,vth,vreset,vleak,τrefre)
const ps_i = S.PSIFNeuron(1,τi,cap,vth,vreset,vleak,τrefri)

# define static inputs
const h_in_e =10.1
const h_in_i =9.0
const in_e = S.IFInputCurrentConstant([h_in_e,])
const in_i = S.IFInputCurrentConstant([h_in_i,])


# connections
const conn_in_e = S.ConnectionIFInput([1.,])
const conn_in_i = S.ConnectionIFInput([1.,])

# connect E <-> I , both ways, but no autapses 
# i connections should be negative!

conn_ie = S.ConnLIF(onesparsemat(1.0))
conn_ei = S.ConnLIF(onesparsemat(-0.8))

# connected populations

pop_e = S.Population(pse1,(conn_ei,conn_in_e),(psi1,in_state_e))
pop_i = S.Population(psi1,(conn_ie,conn_in_i),(pse1,in_state_i))

# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,(pop_e,pop_i))

##

Ttot = 10.
times = (0:myntw.dt:Ttot)
nt = length(times)
# initial conditions
pse1.state_now[1] = v_r
psi1.state_now[1] = v_r + 0.95*(vth-v_r)
# things to save
myvse = Vector{Float64}(undef,nt) # voltage
myfiringe = BitVector(undef,nt) # spike raster
myrefre = similar(myfiringe)  # if it is refractory
eicurr = similar(myvse)  # e-i current 

myvsi = Vector{Float64}(undef,nt)
myfiringi = BitVector(undef,nt)
myrefri = similar(myfiringe)
iecurr = similar(myvsi)

for (k,t) in enumerate(times)
  S.dynamics_step!(t,myntw)
  myvse[k] = pse1.state_now[1]
  myfiringe[k]=pse1.isfiring[1]
  myrefre[k]=pse1.isrefractory[1]
  myvsi[k] = psi1.state_now[1]
  myfiringi[k]=psi1.isfiring[1]
  myrefri[k]=psi1.isrefractory[1]
  eicurr[k]=conn_ei.post_current[1]
  iecurr[k]=conn_ie.post_current[1]
end

# add spikes for plotting purposes, the eight is set arbitrarily to 
# three times the firing threshold
myvse[myfiringe] .= 3 * e1.v_threshold
myvsi[myfiringi] .= 3 * e1.v_threshold;


##

plt=plot(times,myvse;leg=false,linewidth=3,
  ylabel="E (mV)",
  color=colorant"Midnight Blue")

# the green line indicates when the neuron is refractory
plot!(plt,times, 20.0*myrefre; opacity=0.6, color=:green)  

plti=plot(times,myvsi;leg=false,linewidth=3,
   ylabel="I (mV)",
   color=colorant"Brick Red")

plot!(plti,times, 20.0*myrefri; opacity=0.6, color=:green)

pltcurr=plot(times, [ iecurr eicurr]; leg=false, 
  linewidth=3.0, ylabel="E/I connection curr.",
  color=[colorant"Teal" colorant"Orange"])

plot(plt,plti,pltcurr; layout=(3,1),
  xlabel=["" "" "time (s)"])