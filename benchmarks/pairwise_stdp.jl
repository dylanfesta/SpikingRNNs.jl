
using LinearAlgebra,Statistics,StatsBase,Distributions
#using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using Test
using BenchmarkTools
using ProgressMeter

using InvertedIndices

##

function add_fake_spikes!(v::BitArray{1},p::Real)
  for i in eachindex(v)
    v[i] = rand() < p
  end
  return nothing 
end

##
const npre = 2_000 # pre pop 
const npost = 1_500 # post pop 

const pspre = S.PSPoisson(0,0.0,npre)
const pspost = S.PSPoisson(0,0.0,npost)

const Δtplast = 2E-3

const myτplus = 20E-3
const myτminus = 40E-3
const myAplus = 1.0E-3
const myAminus = -0.5E-3
const myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,npost,npre)
const myplasticityT = S.PairSTDPFastT(myτplus,myτminus,myAplus,myAminus,npost,npre)


const dt = 0.1E-3
const wstart = fill(100.0,npost,npre)

##

# _ = let  myconnection =S.ConnectionPlasticityTest(wstart,myplasticityT)
#   add_fake_spikes!(pspost.isfiring,0.3)
#   add_fake_spikes!(pspre.isfiring,0.1)
#   S.plasticity_update!(0.0,dt,pspost,myconnection,pspre,myplasticityT)
# end

##

println("\n")
@info "Benchmark 1 (standard) : \n"
b1 = @benchmark S.plasticity_update!(0.0,$dt,$pspost,myconnection,$pspre,$myplasticity) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticity);
  add_fake_spikes!(pspost.isfiring,0.3);
  add_fake_spikes!(pspre.isfiring,0.1));
show(stdout, MIME("text/plain"), b1)


println("\n")
@info "Benchmark 1 (fast + multithreading) : \n"
b3 = @benchmark S.plasticity_update!($dt,$dt,$pspost,myconnection,$pspre,$myplasticityT) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticityT);
  add_fake_spikes!(pspost.isfiring,0.3);
  add_fake_spikes!(pspre.isfiring,0.1))
show(stdout, MIME("text/plain"), b3)
println("\n")
