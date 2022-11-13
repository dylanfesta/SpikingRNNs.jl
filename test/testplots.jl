using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using Test
using BenchmarkTools
using ProgressMeter
import FileIO: save # save raster png

using InvertedIndices

#using Random; Random.seed!(0)


function onesparsemat(w::Real)
  return sparse(cat(w;dims=2))
end

function wstick(wee::M,wie::M,wei::M,wii::M) where {R<:Real,M<:AbstractMatrix{R}}
  return Matrix(hcat(vcat(wee,wie), (-1).*abs.(vcat(wei,wii) )))
end

function rates_analytic(W::Matrix{R},h::Vector{R}) where R
  return (I-W)\h
end


##

using Plots ; theme(:dark)

##
# Test plasticity rule, both normal and fast version

function oneDSparse(w::Real)
  return sparse(cat(w;dims=2))
end

function post_pre_spiketrains(rate::R,Δt_ro::R,Ttot::R;
    tstart::R = 0.05) where R
  post = collect(range(tstart,Ttot; step=inv(rate)))
  pre = post .- Δt_ro
  return [pre,post] 
end


function post_pre_network(rate::Real,nreps::Integer,Δt_ro::Real,connection::S.Connection)
  Ttot = nreps/rate
  trains = post_pre_spiketrains(rate,Δt_ro,Ttot) 
  ps1 = S.PSFixedSpiketrain(trains[1:1])
  ps2 = S.PSFixedSpiketrain(trains[2:2])
  pop1 = S.UnconnectedPopulation(ps1)
  pop2 = S.Population(ps2,(connection,ps1))
  myntw = S.RecurrentNetwork(dt,pop1,pop2);
  return ps1,ps2,myntw
end

##
const dt = 0.1E-3
# neuron 1 receives no connections
# neuron 2 is connected to 1

const myplasticity = let τplus = 10E-3,
  τminus = 20E-3,
  Aplus = 1E-1,
  Aminus = -0.5E-1
  S.PairSTDP(τplus,τminus,Aplus,Aminus,1,1)
end

const wstart = 100.
const conn_2_1 = S.ConnectionPlasticityTest(oneDSparse(wstart),myplasticity)

const myrate = 0.1
const nreps = 1000
const Ttot = nreps/myrate
const ps1,ps2,myntw = post_pre_network(myrate,nreps,0.1,conn_2_1)

S.reset!(conn_2_1)


function test_stpd_symmetric_rule(rate::R,
    Δt_boundary::R,Ntot::Integer,connection::H.Connection;
    wstart=100.0) where R
  num_spikes = 2*Ntot 
  Δts,population = post_pre_population_morerandom(rate,Δt_boundary,Ntot,connection)
  network = H.RecurrentNetworkExpKernel(population)
  wmat = connection.weights
  ws = Vector{Float64}(undef,num_spikes)
  fill!(wmat,wstart)
  wmat[diagind(wmat)] .= 0.0
  t_now = 0.0
  H.reset!.((network,connection)) # clear spike trains etc
  for k in 1:num_spikes
    t_now = H.dynamics_step_singlepopulation!(t_now,network)
    ws[k] = wmat[1,2]
  end
  Δws = diff(ws)[1:2:end]
  return Δts , ws, Δws
end

## #src

myτplus = 10E-3
myτminus = 30E-3
myAplus = 1E-1
myAminus = -0.5E-1

function expected_symm_stdp(Δt::Real)
  return myAplus*exp(-abs(Δt)/myτplus) + myAminus*exp(-abs(Δt)/myτminus)
end

connection_test = let wmat =  fill(100.0,2,2)
  wmat[diagind(wmat)] .= 0.0
  npost,npre = size(wmat)
  stdp_plasticity = H.SymmetricSTDP(myτplus,myτminus,myAplus,myAminus,npost,npre)
  H.ConnectionWeights(wmat,stdp_plasticity)
end

theplot = let myrate = 0.1
  mybound = 100E-3
  myNtest  = 800
  xplot = range(-mybound,mybound,length=150)
  delta_ts,testws, testDws = test_stpd_symmetric_rule(myrate,mybound,myNtest,connection_test)
  scatter(delta_ts,testDws, xlabel="Delta t",ylabel="dw/dt", 
    label="numeric", title="Symmetric STDP")
  plot!(xplot, expected_symm_stdp.(xplot),linewidth=2,linestyle=:dash,color=:red,
    label="analytic")
end


# **THE END**

using Literate; Literate.markdown("examples/plasticity_STDP.jl","docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src

  function IFInputSpikesTrain(train::Vector{Vector{Float64}})
