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

function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end


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
end

function expected_pairwise_stdp(Δt::R,τplus::R,τminus::R,Aplus::R,Aminus::R) where R
  return Δt > 0 ? Aplus*exp(-Δt/τplus) : Aminus*exp(Δt/τminus)
end

##

Δtplast = 2E-3
Δttest = rand(Uniform(-0.1,0.1),40)
dt = 0.5E-3
myτplus = 10E-3
myτminus = 45E-3
myAplus = 1.0
myAminus = -0.5
myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,1,1)
myplasticityF = S.PairSTDPFast(Δtplast,myτplus,myτminus,myAplus,myAminus,1,1)
my_connection = S.ConnectionPlasticityTest(oneDSparse(100.0),myplasticity)
my_connectionF = S.ConnectionPlasticityTest(oneDSparse(100.0),myplasticityF)
myrate = 0.1
nreps = 10
@time Δws_num = map( Δt -> test_stpd_rule(Δt,myrate,nreps,my_connection), Δttest)
@time Δws_numF = map( Δt -> test_stpd_rule(Δt,myrate,nreps,my_connectionF), Δttest)
Δws_an = map( Δt -> expected_pairwise_stdp(Δt,myτplus,myτminus,myAplus,myAminus), Δttest)


plotvs(Δws_num,Δws_an)
plotvs(Δws_numF,Δws_an)
plotvs(Δws_numF,Δws_num)


@test all(isapprox.(Δws_num,Δws_an;atol=1E-2))

@show extrema(Δws_num .- Δws_an)

error()

##
const dt = 0.1E-3

const myτplus = 10E-3
const myτminus = 45E-3
const myAplus = 1E-1
const myAminus = -0.5E-1


const myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,1,1)

const myrate = 0.5
const nreps = 10
const my_connection = S.ConnectionPlasticityTest(oneDSparse(100.0),myplasticity)
const Δt_test = range(-0.2,0.2,length=100)

const Δws = @showprogress map(Δt_test) do Δt
  test_stpd_rule(Δt,myrate,nreps,my_connection)
end


theplot = let Δtplot = range(extrema(Δt_test)...,length=200)
  plt = scatter(Δt_test,Δws;label="simulation")
  plot!(plt,Δtplot,expected_stdp.(Δtplot); label="expected",
   xlabel="Δt",ylabel="Δw",title="STDP rule")
end;
plot(theplot)



## #src