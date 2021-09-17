push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using FFTW

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

##

dt = 1E-3
Ttot = 10.0
# One LIF neuron
myτ = 0.2
vth = 10.
v_r = -5.0
τrefr= 0.3 # refractoriness
τpcd = 0.2 # post synaptic current decay
myinput = 0.0 # constant input to E neuron
ps_e = S.PSLIF(myτ,vth,v_r,τrefr,τpcd,1)

# one static input 
in_state_e = S.PSSimpleInput(S.InputSimpleOffset(myinput))
# connection will be FakeConnection()

# let's produce a couple of trains
train1 = let rat = 1.0
  sort(rand(Uniform(0.05,Ttot),round(Integer,rat*Ttot) ))
end
train2 = let rat = 0.5
  sort(rand(Uniform(0.05,Ttot),round(Integer,rat*Ttot) ))
end
# input population
ps_train_in=S.PSFixedSpiketrain([train1,train2],myτ)

# and connection object
conn_e_in = let w_intrain2e = sparse([eps() Inf ; ])
  S.ConnSpikeTransfer(w_intrain2e)
end

# connected populations
# two populations: the input population (unconnected) 
# and the E neuron connected to input
pop_in = S.UnconnectedPopulation(ps_train_in)
pop_e = S.Population(ps_e,(conn_e_in,ps_train_in),
  (S.FakeConnection(),in_state_e))


##
# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop_in,pop_e)

# record spiketimes and internal potential
krec = 1
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)
rec_spikes_e = S.RecSpikes(ps_e,100.0,Ttot)
rec_spikes_in = S.RecSpikes(ps_train_in,100.0,Ttot)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e,rec_spikes_in])
S.reset!.([ps_e,ps_train_in])
# initial conditions
ps_e.state_now[1] = 0.0

for (k,t) in enumerate(times)
  rec_state_e(t,k,myntw)
  rec_spikes_e(t,k,myntw)
  rec_spikes_in(t,k,myntw)
  S.dynamics_step!(t,myntw)
end

S.add_fake_spikes!(1.5vth,rec_state_e,rec_spikes_e)
##



plot(rec_state_e.times,rec_state_e.state_now[:];linewidth=3,leg=false)

spkt,spkneu = S.get_spiketimes_spikeneurons(rec_spikes_e)
scatter(spkt,spkneu;leg=false,marker=:vline,markersize=10.)

_ = let  (spkt,spkneu) = S.get_spiketimes_spikeneurons(rec_spikes_in)
  scatter(spkt,spkneu;leg=false,marker=:vline,markersize=10.)
end

train1_sim,train2_sim = let (spkt,spkneu) = S.get_spiketimes_spikeneurons(rec_spikes_in)
  spkt[spkneu .== 1],spkt[spkneu .== 2]
end

traine_sim = let (spkt,spkneu) = S.get_spiketimes_spikeneurons(rec_spikes_e)
  spkt
end


@test all(isapprox.(train1,train1_sim;atol=1.1dt))
@test all(isapprox.(train2,train2_sim;atol=1.1dt))
@test all(isapprox.(traine_sim,sort(vcat(train1_sim,train2_sim));atol=1.1dt))

##
dt = 5E-4
Ttot = 3.0 
myτ = 0.1
vth = 12.
v_r = -6.123
τrefr = 0.0
τpcd = 1E10
myinput = 14.0
ps_e = S.PSLIF(myτ,vth,v_r,τrefr,τpcd,1)
# create static input 
in_state_e = S.PSSimpleInput(S.InputSimpleOffset(myinput))
# only one population: E with input
pop_e = S.Population(ps_e,(S.FakeConnection(),in_state_e))
# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop_e)

times = (0:myntw.dt:Ttot)
nt = length(times)
S.expected_period_norefr(ps_e.neurontype,myinput)
# spike recorder
rec_spikes = let exp_freq = inv(S.expected_period_norefr(ps_e.neurontype,myinput))
  S.RecSpikes(ps_e,1.5*exp_freq,Ttot)
end
# reset and run 
S.reset!(rec_spikes)
S.reset!(ps_e)
# initial conditions
ps_e.state_now[1] = v_r

# run!
for (k,t) in enumerate(times)
  rec_spikes(t,k,myntw)
  S.dynamics_step!(t,myntw)
end
##
spkt,_ = S.get_spiketimes_spikeneurons(rec_spikes)
# period of first spike
@test isapprox(S.expected_period_norefr(ps_e.neurontype,myinput),spkt[1] ;
  atol = 0.02)

# number of spikes
myper_postrest = S.expected_period_norefr(e1.τ,0.0,e1.v_threshold,my_input)
nspk_an = floor(Ttot/(e1.τ_refractory + myper_postrest ) )
@test isapprox(nspk_an,count(myfiring) ; atol=2)