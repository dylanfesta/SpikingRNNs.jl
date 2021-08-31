push!(LOAD_PATH, abspath(@__DIR__,".."))
using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

##
dt = 1E-3
# two LIF neurons, E and I
myτe = 0.2
myτi = 0.1
vth = 10.
v_r = -5.0
τrefre = 0.3 # refractoriness
τrefri = 0.1 
τpcd = 0.2 # post synaptic current decay
e1 = S.NTLIF(myτe,vth,v_r,τrefre,τpcd)
i1 = S.NTLIF(myτi,vth,v_r,τrefri,τpcd)
pse1 = S.PSLIF(e1,1)
psi1 = S.PSLIF(i1,1)

# one static input 
h_in_e = onesparsemat(10.1)
h_in_i = onesparsemat(9.0)
in_type = S.InputSimpleOffset()
in_state_e = S.PSSimpleInput(in_type)
in_state_i = S.PSSimpleInput(in_type)

# connect E <-> I , both ways, but no autapses 
# i connections should be negative!

conn_ie = S.ConnLIF(onesparsemat(1.0))
conn_ei = S.ConnLIF(onesparsemat(-0.8))
conn_in_e = S.BaseConnection(h_in_e)
conn_in_i = S.BaseConnection(h_in_i)

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