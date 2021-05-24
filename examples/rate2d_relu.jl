# created: 20210519 
# 2D rate model with balanced amplification effect. 

push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
using Test
pkg"activate ."

using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs


##

ne = 1
ni = 1

τe = 1.
τi = 1.

α = 1.

# populations
pope = S.PopRateReLU(ne,τe,α)
popi = S.PopRateReLU(ni,τi,α)

pse = S.PSRate(pope)
psi = S.PSRate(popi)
S.reset_input!.((pse,psi))


# connections
function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

(w_ee,w_ie,w_ei,w_ii) = let w = 20. , k = 1.1
  onesparsemat.((w,w,-k*w,-k*w))
end

conn_ee = S.ConnectionRate(pse,w_ee,pse)
conn_ei = S.ConnectionRate(pse,w_ei,psi)
conn_ie = S.ConnectionRate(psi,w_ie,pse)
conn_ii = S.ConnectionRate(psi,w_ii,psi)

# inputs
in_e = S.PopInputStatic(pse,[0.,])
in_i = S.PopInputStatic(psi,[0.,])

# initial conditions
pse.state_now[1] = S.ioinv(10.0,pse)
psi.state_now[1] = S.ioinv(5.0,pse)


dt = 1E-4
T = 10.0
times = 0:dt:T 
ntimes = length(times)
mynetwork = S.RecurrentNetwork(dt,(pse,psi),(in_e,in_i),
  (conn_ee,conn_ie,conn_ei,conn_ii) )


e_out = Vector{Float64}(undef,ntimes)
i_out = Vector{Float64}(undef,ntimes)

##


for (k,t) in enumerate(times) 
  e_out[k] = S.iofunction(pse.state_now[1],pse)
  i_out[k] = S.iofunction(psi.state_now[1],psi)
  # rate model with constant input  does not really depend on absolute time (first argument)
  S.dynamics_step!(0.0,mynetwork)
end

plot(times,[e_out i_out];linewidth=4,leg=false,
  color=[colorant"Teal" colorant"Salmon"])
