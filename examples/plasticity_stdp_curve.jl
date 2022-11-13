#=
# Plasticity: classic STDP protocol

Plasticity between 2 "virtual" neurons that always fire at fixed $\Delta t$ intervals.
=#

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
function oneDSparse(w::Real)
  return sparse(cat(w;dims=2))
end

## #src
# # Utility functions here

# This function produces 2 spiketrains. Each spiketrain has the given rate, and the spikes
# are offset by $\Delta t$.
function post_pre_spiketrains(rate::R,Δt_ro::R,Ttot::R;
    tstart::R = 0.05) where R
  post = collect(range(tstart,Ttot; step=inv(rate)))
  pre = post .- Δt_ro
  return [pre,post] 
end;

# This function generates virtual neuron and network object
function post_pre_network(rate::Real,nreps::Integer,Δt_ro::Real,connection::S.Connection)
  Ttot = nreps/rate
  trains = post_pre_spiketrains(rate,Δt_ro,Ttot) 
  ps1 = S.PSFixedSpiketrain(trains[1:1])
  ps2 = S.PSFixedSpiketrain(trains[2:2])
  pop1 = S.UnconnectedPopulation(ps1)
  pop2 = S.Population(ps2,(connection,ps1))
  myntw = S.RecurrentNetwork(dt,pop1,pop2);
  return ps1,ps2,myntw
end;

# This applies the plasticity rule to the network, and returns the difference in weight
function test_stpd_rule(Δt::R,rate::R,
    nreps::Integer,connection::S.ConnectionPlasticityTest;wstart=100.0) where R
  fill!(connection.weights,wstart)
  myntw = post_pre_network(rate,nreps,Δt,connection)[3]
  S.reset!(connection)
  Ttot = (nreps+2)/myrate
  times = (0:dt:Ttot)
  for t in times 
    S.dynamics_step!(t,myntw)
  end
  Δw = (connection.weights[1,1] - wstart)/nreps
  return Δw
end;

# Finally, this is the STDP curve that we would expect analytically
function expected_pairwise_stdp(Δt::R,τplus::R,τminus::R,Aplus::R,Aminus::R) where R
  return Δt > 0 ? Aplus*exp(-Δt/τplus) : Aminus*exp(Δt/τminus)
end;

## #src
# # Set some constants and run STDP

const dt = 0.1E-3

const myτplus = 10E-3
const myτminus = 45E-3
const myAplus = 1E-1
const myAminus = -0.5E-1;

# This line defines the plasticity that is being tested
# The plasticity goes into the connection object
const myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,1,1);
# `ConnectionPlasticityTest` is a special type: it is subject to plasticity rules,
# but neurons do not exchange signals through it.
const my_connection = S.ConnectionPlasticityTest(oneDSparse(100.0),myplasticity)

const myrate = 0.5
const nreps = 10
const Δt_test = range(-0.2,0.2,length=100)

# Here is the main computation:
const Δws = map(Δt_test) do Δt
  test_stpd_rule(Δt,myrate,nreps,my_connection)
end;

# Plot the result:
theplot = let Δtplot = range(extrema(Δt_test)...,length=200)
  plt = scatter(Δt_test,Δws;label="simulation")
  plot!(plt,Δtplot,expected_pairwise_stdp.(Δtplot,myτplus,myτminus,myAplus,myAminus);
   xlabel="Δt",ylabel="Δw",title="STDP rule", linewidth=2, linestyle=:dash, label="analytic") 
end;
plot(theplot)


# **The end!**

## Publish ! #src
using Literate #src
Literate.markdown(@__FILE__,"docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/SpikingRNNs.jl/blob/master") #src
