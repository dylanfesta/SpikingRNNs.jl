push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using ProgressMeter

function onesparsemat(w::Real)
  return sparse(fill(w,(1,1)))
end

using Random; Random.seed!(0)


##

const Ttot = 1.0
const dt = 0.1E-3
const Nneus = 500
const rates = rand(Uniform(20.,50.),Nneus)

ps_all = let ratefun(_) = rates 
  τ = 1E6
  S.PSInputPoissonFtMulti(ratefun,τ,Nneus)
end

myplasticity = let νdeath = 1.0,
  Δt = Ttot/200,
  ρ = 0.6,
  w_start = 1.0
  S.PlasticityStructural(νdeath,Δt,ρ,w_start;no_autapses=true)
end

weights_start = let ρstart = 0.1
  S.sparse_constant_wmat(Nneus,Nneus,ρstart,2.0;no_autapses=true)
end

conn_all=S.ConnectionPlasticityTest(copy(weights_start),myplasticity)

pop_all=S.Population(ps_all,(conn_all,ps_all))

myntw = S.RecurrentNetwork(dt,pop_all)

rec_spikes_all = S.RecSpikes(ps_all,1.5*maximum(rates),Ttot;idx_save=collect(1:100))
# must save ALL weights, even empty ones
rec_weights = S.RecWeights(conn_all,5,dt,Ttot; 
  idx_save=CartesianIndices(weights_start)[:])


## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# reset weights
copy!(conn_all.weights,weights_start)
# clean up
S.reset!(rec_spikes_all)
S.reset!(ps_all)
S.reset!(conn_all)

@time begin
  @showprogress 1.0 "network simulation " for (k,t) in enumerate(times)
    rec_spikes_all(t,k,myntw)
    rec_weights(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
end

##

wtimes,weights_out = S.rebuild_weights(rec_weights)

##
heatmap(Array(weights_out[1]))
heatmap(Array(weights_out[end]))

##
# connectoon density change

function conndens(w::SparseMatrixCSC)
  return nnz(w) / ( (*)(size(w)...))
end

cdenss = conndens.(weights_out)

plot(wtimes,cdenss;leg=false)
##