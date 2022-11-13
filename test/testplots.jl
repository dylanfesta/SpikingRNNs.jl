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

const n1 = 13 # pre pop 
const n2 = 7 # post pop 

# const ps1 = let rates = rand(Uniform(3.,20.),n1)
#   ratefun(t) = rates
#   S.PSInputPoissonFtMulti(ratefun,0.0,n1)
# end
# const ps2 = let rates = rand(Uniform(7.,12.),n2)
#   ratefun(t) = rates
#   S.PSInputPoissonFtMulti(ratefun,0.0,n2)
# end

# this is here mostly for testing
function make_poisson_samples(rate::R,t_tot::R) where R
  ret = Vector{R}(undef,round(Integer,1.3*rate*t_tot+10)) # preallocate
  t_curr = zero(R)
  k_curr = 1
  while t_curr <= t_tot
    Δt = -log(rand())/rate
    t_curr += Δt
    ret[k_curr] = t_curr
    k_curr += 1
  end
  return keepat!(ret,1:k_curr-2)
end

const Ttot = 20.0
const ps1 = let rates = rand(Uniform(3.,20.),n1)
  trains = make_poisson_samples.(rates,Ttot)
  S.PSFixedSpiketrain(trains)
end
const ps2 = let rates = rand(Uniform(7.,12.),n2)
  trains = make_poisson_samples.(rates,Ttot)
  S.PSFixedSpiketrain(trains)
end


const Δtplast = 2E-3
const dt = 0.1E-3

const myτplus = 20E-3
const myτminus = 40E-3
const myAplus = 1.0E-3
const myAminus = -0.5E-3
const myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,n2,n1)
const myplasticityF = S.PairSTDPFast(Δtplast,myτplus,myτminus,myAplus,myAminus,n2,n1)

const wstart = fill(100.0,n2,n1)
const my_connection = S.ConnectionPlasticityTest(wstart,myplasticity)
const my_connectionF = S.ConnectionPlasticityTest(wstart,myplasticityF)

const pop1 = S.UnconnectedPopulation(ps1)
const pop2 = S.Population(ps2,(my_connection,ps1))
const pop2F = S.Population(ps2,(my_connectionF,ps1))
const myntw = S.RecurrentNetwork(dt,pop1,pop2)
const myntwF = S.RecurrentNetwork(dt,pop1,pop2F)

const  times = (0:dt:Ttot)
S.reset!.((ps1,ps2,my_connection,my_connectionF))
fill!(my_connection.weights,100.0)
fill!(my_connectionF.weights,100.0)

@time for t in times 
  S.dynamics_step!(t,myntw)
end
Δw = my_connection.weights .- 100.0

S.reset!.((ps1,ps2,my_connection,my_connectionF))
fill!(my_connection.weights,100.0)
fill!(my_connectionF.weights,100.0)

@time for t in times 
  S.dynamics_step!(t,myntwF)
end
ΔwF = my_connectionF.weights .- 100.0

@test all(isapprox.(Δw,ΔwF,atol=1E-3))

plotvs(Δw,ΔwF)