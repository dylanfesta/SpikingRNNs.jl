using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using Test
using BenchmarkTools
using ProgressMeter
import FileIO: save # save raster png

using InvertedIndices

using Random; Random.seed!(0)


function onesparsemat(w::Real)
  return sparse(cat(w;dims=2))
end
##

using Plots ; theme(:dark)

##

# test 1, unconnected Poisson neurons
# check rate, and FF

N = 50
dt = 0.1E-3
τ = 50E-3
h_in = 123.45

# population
ps =  S.PSPoissonNeuron(τ,N)
# input
ps_in = S.PoissonInputCurrentConstant(fill(h_in,N))
# non-existing recurring connection
conn_ee = S.ConnectionPoissonExpKernel(S.PoissonExcitatory(),-1E6,fill(0.0,N,N))

# population
pop = S.Population(ps,(conn_ee,ps),(S.FakeConnection(),ps_in))

# network
ntw = S.RecurrentNetwork(dt,pop)

Ttot = 60.0
# record spiketimes and internal potential
rec_spikes_e = S.RecSpikes(ps,200.0,Ttot)

## Run

times = (0:ntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([ps,rec_spikes_e])
# initial conditions
ps.state_now .= 30.0

@time begin
  @showprogress 5.0 "network simulation " for (k,t) in enumerate(times)
    rec_spikes_e(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
end

spikes_c = S.get_content(rec_spikes_e)

st,sn=S.get_spiketimes_spikeneurons(spikes_c)
# myrast = S.raster_png(0.01,rec_spikes_e;Nneurons=Ne,Ttot=Ttot)
# save("/tmp/rast.png",myrast)
