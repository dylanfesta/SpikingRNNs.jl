#=
Goal : compare neuron-like input with fake connection input
type, make sure they match perfectly 
=#

push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using ProgressMeter
import FileIO: save # save raster png

using InvertedIndices

using Random; Random.seed!(0)

##

# define E population of conductance based EIF neurons
const dt = 0.1E-3
const Ttot = 1.0
const Ne = 5

myτe = 20E-3 # seconds
τrefr= 15E-3 # refractoriness
vth_e = 20.   # mV
vthexp = -50.0 # actual threshold for spike-generation
eifslope = 2.0
v_rest_e = -70.0
v_rev_e = 0.0
v_leak_e = v_rest_e
v_reset_e = v_rest_e
Cap = 300.0 #capacitance mF

myτe_ker = 20E-3 # synaptic kernel 
myτe_ker_plus = 8E-3
myτe_ker_minus = 3E-3

# synaptic_kernel = S.SKExp(myτe_ker)
synaptic_kernel = S.SKExpDiff(myτe_ker_plus,myτe_ker_minus)

nt_e,ps_e = let sgen = S.SpikeGenEIF(vthexp,eifslope)
  nt = S.NTLIFConductance(synaptic_kernel,sgen,myτe,Cap,
    vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
  ps = S.PSLIFConductance(nt,Ne)
  (nt,ps)
end

## define input trains
const in_rate = 30.0
const w_in_vec = rand(Uniform(80.0,200.0),Ne) 
onetrain(rate,Ttot) = sort(rand(Uniform(0,Ttot),round(Integer,rate*Ttot)))
trains = [ onetrain(in_rate,Ttot) for _ in 1:Ne ]
spkgen = S.SGTrains(trains)

nt_in_type1 = S.NTInputConductance(spkgen,synaptic_kernel,v_rev_e)
ps_in_type1 = S.PSInputConductance(nt_in_type1,Ne)

nt_in_type2 = S.NTInputConductance(spkgen,synaptic_kernel,v_rev_e)
ps_in_type2 = S.PSInputPoissonConductance(nt_in_type2,w_in_vec)

## define connection for type 1 input
conn_e_in = let wmat=sparse(Diagonal(w_in_vec))
  S.ConnGeneralIF2(wmat)
end

## connect !
pop_et1 = S.Population(ps_e,(conn_e_in,ps_in_type1))
pop_in1 = S.Population(ps_in_type1,(S.InputDummyConnection(),ps_in_type1))
pop_et2 = S.Population(ps_e,(S.InputDummyConnection(),ps_in_type2))

ntw1 = S.RecurrentNetwork(dt,pop_et1,pop_in1)
ntw2 = S.RecurrentNetwork(dt,pop_et2)

## initial conditions
initial_states = rand(Uniform(v_reset_e,-49.0),Ne)

##
const krec = 1
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)
rec_spikes = S.RecSpikes(ps_e,50.0,Ttot)

## Run
times = (0:dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes])
S.reset!.([ps_e,ps_in_type1,ps_in_type2])
S.reset!(conn_e_in)
# initial conditions
ps_e.state_now .= initial_states

testinput1=Vector{Float64}[]
@time begin
  @showprogress 1.0 "network simulation " for (k,t) in enumerate(times)
    rec_state_e(t,k,ntw1)
    rec_spikes(t,k,ntw1)
    push!(testinput1,ps_e.input)
    S.dynamics_step!(t,ntw1)
  end
end
S.add_fake_spikes!(vth_e,rec_state_e,rec_spikes)

testinput1=hcat(testinput1...)

## plot the state for one neuron

_ = let neu = 4,
  plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[neu,:];linewidth=2,leg=false,
    xlims=(0,1))
end

spkdic=S.get_spiketimes_dictionary(rec_spikes)

## now try the other one

rec_state_e2 = S.RecStateNow(ps_e,krec,dt,Ttot)
rec_spikes2 = S.RecSpikes(ps_e,50.0,Ttot)

## Run
# clean up
S.reset!.([rec_state_e2,rec_spikes2])
S.reset!.([ps_e,ps_in_type1,ps_in_type2])
# initial conditions
ps_e.state_now .= initial_states

testinput2=Vector{Float64}[]

@time begin
  @showprogress 1.0 "network simulation " for (k,t) in enumerate(times)
    rec_state_e2(t,k,ntw2)
    rec_spikes2(t,k,ntw2)
    push!(testinput2,ps_e.input)
    S.dynamics_step!(t,ntw2)
  end
end
S.add_fake_spikes!(vth_e,rec_state_e2,rec_spikes2)
testinput2=hcat(testinput2...)
## plot the state for one neuron

_ = let neu = 4,
  plt=plot()
  plot!(plt,rec_state_e2.times,rec_state_e2.state_now[neu,:];linewidth=2,leg=false)
  plot!(plt,rec_state_e.times,rec_state_e.state_now[neu,:];linewidth=2,leg=false,
    linestyle=:dash)
end

#spkdic=S.get_spiketimes_dictionary(rec_spikes2)