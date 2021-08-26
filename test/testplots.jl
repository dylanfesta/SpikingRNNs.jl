push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) ; plotlyjs()
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using FFTW

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

##

neuron_ei = S.NTReQuadratic(1.0,0.01)
psei  = S.PSRate(neuron_ei,2)

wmat = sparse([ 2.  -3.
                2.5  -0.5 ]) 
input_mat = let ret=Matrix{Float64}(undef,2,1)
  ret.=[50.33,2.8]
  sparse(ret)
end

conn_rec = S.BaseFixedConnection(neuron_ei,wmat,neuron_ei)                  

fpoint = - inv(Matrix(wmat)-I)*input_mat
@info fpoint
## input connection!

in_type = S.InputSimpleOffset()
in_state = S.PSSimpleInput(in_type)
conn_in = S.BaseFixedConnection(neuron_ei,input_mat,in_state)
##

pop_ei = S.Population(psei,(conn_rec,conn_in),(psei,in_state))



##

dt = 1E-2
T = 60.0
times = 0:dt:T 
ntimes = length(times)
mynetwork = S.RecurrentNetwork(dt,(pop_ei,))

##

ei_out = Matrix{Float64}(undef,2,ntimes)
# initial conditions
psei.state_now .= S.ioinv(10.0,psei)

for (k,t) in enumerate(times) 
  ei_out[:,k] = S.iofunction.(psei.state_now,neuron_ei)
  # rate model with constant input  does not really depend on absolute time (first argument)
  S.dynamics_step!(mynetwork)  
end


##
plot(times[1:20:end],ei_out[1,1:20:end];leg=false)
plot!(times[1:20:end],ei_out[2,1:20:end];leg=false)

##

ei_out[:,end]

fpoint

ei_out[:,end] .- fpoint

##

struct NTPoissonCO 
  rate::Ref{Float64} # rate is in Hz and is mutable
  τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
  v_reversal::Float64 # reversal potential that affects postsynaptic neurons
end

##

boh=NTPoissonCO(10.,2.,3.)

##
wtest = S.sparse_wmat_lognorm(100,200,0.1,3.,2.0;rowsum=3.)
plt=histogram(nonzeros(wtest);nbins=30)


mean(nonzeros(wtest))

sum(wtest;dims=2)