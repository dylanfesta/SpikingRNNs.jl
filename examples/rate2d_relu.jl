# created: 20210519 
# 2D linear rate model with balanced amplification effect. 

push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs


# Parameters

ne = 1
ni = 1

τe = 1.
τi = 1.

α = 1.

# Define neuron types and population states

neuron_e = S.NTReLU(τe,α)
neuron_i = S.NTReLU(τi,α)
pse  = S.PSRate(neuron_e,ne)
psi  = S.PSRate(neuron_i,ni)

# Define the connections
function onesparsemat(w::Real)
  mat = Matrix{Float64}(undef,1,1)
  mat[1,1] = w
  return sparse(mat)
end

(w_ee,w_ie,w_ei,w_ii) = let w = 20. , k = 1.1
  onesparsemat.((w,w,-k*w,-k*w))
end

conn_ee = S.BaseFixedConnection(neuron_e,w_ee,neuron_e)
conn_ei = S.BaseFixedConnection(neuron_e,w_ei,neuron_i)
conn_ie = S.BaseFixedConnection(neuron_i,w_ie,neuron_e)
conn_ii = S.BaseFixedConnection(neuron_i,w_ii,neuron_i)

# inputs
h_in_e = onesparsemat(0.0)
h_in_i = onesparsemat(0.0)
in_state_e = S.PSSimpleInput(S.InputSimpleOffset())
in_state_i = S.PSSimpleInput(S.InputSimpleOffset())
conn_in_e = S.BaseFixedConnection(neuron_e,h_in_e,in_state_e)
conn_in_i = S.BaseFixedConnection(neuron_i,h_in_i,in_state_i)

# populations are population states, plus all incoming connections
# plus presynaptic population states

pop_e = S.Population(pse,(conn_ee,conn_ei,conn_in_e),(pse,psi,in_state_e))
pop_i = S.Population(psi,(conn_ie,conn_ii,conn_in_i),(pse,psi,in_state_i))

dt = 1E-4
T = 10.0
times = 0:dt:T 
ntimes = length(times)
mynetwork = S.RecurrentNetwork(dt,(pop_e,pop_i))


e_out = Vector{Float64}(undef,ntimes)
i_out = Vector{Float64}(undef,ntimes)

##

# initial conditions
pse.state_now[1] = S.ioinv(10.0,pse)
psi.state_now[1] = S.ioinv(5.0,pse)


for (k,t) in enumerate(times) 
  e_out[k] = S.iofunction(pse.state_now[1],pse)
  i_out[k] = S.iofunction(psi.state_now[1],psi)
  # rate model with constant input  does not really depend on absolute time (first argument)
  S.dynamics_step!(0.0,mynetwork)
end

plot(times,[e_out i_out];linewidth=4,leg=:topright,
  color=[colorant"Teal" colorant"Salmon"],label=["exc" "inh"])
