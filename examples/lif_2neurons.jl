push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

using Test
using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools


##
dt = 1E-3
# two LIF neurons, E and I
myτe = 0.2
myτi = 0.1
vth = 10.
v_r = -5.0
τrefre = 0.3
τrefri = 0.1
τpcd = 0.2
e1 = S.PopLIF(1,myτe,vth,v_r,τrefre,τpcd)
i1 = S.PopLIF(1,myτi,vth,v_r,τrefri,τpcd)
pse1 = S.PSLIF(e1)
psi1 = S.PSLIF(i1)

# one static input 
in_e = 12.5
in_i = 10.5
pse_in = S.PopInputStatic(pse1,[in_e,])
psi_in = S.PopInputStatic(psi1,[in_i,])

# connect E <-> I , both ways, but no autapses 
# i connections should be negative!
function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

conn_ie = S.ConnectionLIF(psi1,onesparsemat(1.0),pse1)
conn_ei = S.ConnectionLIF(pse1,onesparsemat(-0.8),psi1)
# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,(pse1,psi1),(pse_in,psi_in),(conn_ie,conn_ei) )

##

Ttot = 10.
times = (0:myntw.dt:Ttot)
nt = length(times)
pse1.state_now[1] = v_r
psi1.state_now[1] = v_r + 0.95*(vth-v_r)
myvse = Vector{Float64}(undef,nt)
myfiringe = BitVector(undef,nt)
myrefre = similar(myfiringe)
myvsi = Vector{Float64}(undef,nt)
myfiringi = BitVector(undef,nt)
myrefri = similar(myfiringe)
eicurr = similar(myvsi)
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

myvse[myfiringe] .= 3 * e1.v_threshold
myvsi[myfiringi] .= 3 * e1.v_threshold;


##

plt=plot(times,myvse;leg=false,linewidth=3,
  ylabel="E (mV)",
  color=colorant"Midnight Blue")

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