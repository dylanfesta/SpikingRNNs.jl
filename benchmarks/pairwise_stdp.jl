
using LinearAlgebra,Statistics,StatsBase,Distributions
#using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using Test
using BenchmarkTools
using ProgressMeter

using InvertedIndices


function add_fake_spikes!(v::BitArray{1},p::Real)
  for i in eachindex(v)
    v[i] = rand() < p
  end
  return nothing 
end

##
npre = 2_000 # pre pop 
npost = 1_500 # post pop 

pspre = S.PSPoisson(0,0.0,npre)
pspost = S.PSPoisson(0,0.0,npost)

myτplus = 20E-3
myτminus = 40E-3
myAplus = 1.0E-3
myAminus = -0.5E-3

dt = 0.1E-3
wstart = let mask = (rand(npost,npre) .< 0.9)
  sparse(fill(100.,npost,npre) .* mask)
end

myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,npost,npre)
myplasticityT = S.PairSTDP_T(myτplus,myτminus,myAplus,myAminus,wstart)



@info "Large size network \n"

@info "Benchmark 1 (standard) : \n"
b1 = @benchmark S.plasticity_update!(0.0,$dt,$pspost,myconnection,$pspre,$myplasticity) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticity);
  add_fake_spikes!(pspost.isfiring,0.3);
  add_fake_spikes!(pspre.isfiring,0.1));
show(stdout, MIME("text/plain"), b1)


println("\n")
@info "Benchmark 1 (fast + multithreading, $(Threads.nthreads()) threads) : \n"
b3 = @benchmark S.plasticity_update!($dt,$dt,$pspost,myconnection,$pspre,$myplasticityT) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticityT);
  add_fake_spikes!(pspost.isfiring,0.3);
  add_fake_spikes!(pspre.isfiring,0.1))
show(stdout, MIME("text/plain"), b3)
println("\n")


@info "Repeat with size 200 network \n"


npre = 200 # pre pop 
npost = 200 # post pop 

pspre = S.PSPoisson(0,0.0,npre)
pspost = S.PSPoisson(0,0.0,npost)


dt = 0.1E-3
wstart = fill(100.0,npost,npre)

myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,npost,npre)
myplasticityT = S.PairSTDP_T(myτplus,myτminus,myAplus,myAminus,wstart)



@info "Benchmark 1 (standard) : \n"
b1 = @benchmark S.plasticity_update!(0.0,$dt,$pspost,myconnection,$pspre,$myplasticity) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticity);
  add_fake_spikes!(pspost.isfiring,0.3);
  add_fake_spikes!(pspre.isfiring,0.1));
show(stdout, MIME("text/plain"), b1)


println("\n")
@info "Benchmark 1 (fast + multithreading, $(Threads.nthreads()) threads) : \n"
b3 = @benchmark S.plasticity_update!($dt,$dt,$pspost,myconnection,$pspre,$myplasticityT) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticityT);
  add_fake_spikes!(pspost.isfiring,0.3);
  add_fake_spikes!(pspre.isfiring,0.1))
show(stdout, MIME("text/plain"), b3)
println("\n")


@info "Repeat with size 500 network and LOW FREQUENCY \n"

freq = 60.0
npre = 500 # pre pop 
npost = 500 # post pop 

pspre = S.PSPoisson(0,0.0,npre)
pspost = S.PSPoisson(0,0.0,npost)

dt = 0.1E-3
wstart = fill(100.0,npost,npre)

myplasticity = S.PairSTDP(myτplus,myτminus,myAplus,myAminus,npost,npre)
myplasticityT = S.PairSTDP_T(myτplus,myτminus,myAplus,myAminus,wstart)


@info "Benchmark 1 (standard) : \n"
b1 = @benchmark S.plasticity_update!(0.0,$dt,$pspost,myconnection,$pspre,$myplasticity) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticity);
  add_fake_spikes!(pspost.isfiring,dt*freq);
  add_fake_spikes!(pspre.isfiring,dt*freq));
show(stdout, MIME("text/plain"), b1)


println("\n")
@info "Benchmark 1 (fast + multithreading, $(Threads.nthreads()) threads) : \n"
b3 = @benchmark S.plasticity_update!($dt,$dt,$pspost,myconnection,$pspre,$myplasticityT) setup=(
  myconnection =S.ConnectionPlasticityTest(wstart,myplasticityT);
  add_fake_spikes!(pspost.isfiring,dt*freq);
  add_fake_spikes!(pspre.isfiring,dt*freq))
show(stdout, MIME("text/plain"), b3)
println("\n")

