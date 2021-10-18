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

# let's make a network with strong E-E connections that explodes

# define E population of conductance based EIF neurons
const dt=0.1E-3
const Ne = 10

myτe = 20E-3 # seconds
τrefr= 15E-3 # refractoriness
vth_e = 20.   # mV
myτe_ker = 10E-3 # synaptic kernel 
vthexp = -50.0 # actual threshold for spike-generation
eifslope = 2.0
v_rest_e = -70.0
v_rev_e = 0.0
v_leak_e = v_rest_e
v_reset_e = v_rest_e
Cap = 300.0 #capacitance mF


nt_e,ps_e = let sker = S.SKExp(myτe_ker)
  sgen = S.SpikeGenEIF(vthexp,eifslope)
  nt = S.NTLIFConductance(sker,sgen,myτe,Cap,
    vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
  ps = S.PSLIFConductance(nt,Ne)
  (nt,ps)
end

## define E to E connections
const jee = 0.1
const pee = 0.1
wmat = S.sparse_constant_wmat(Ne,Ne,pee,jee;no_autapses=true)
conn_e_e = S.ConnGeneralIF2(wmat)

## define external input of Poisson neurons
const ext_rate = 30.0
const je_in = 1000.0
nt_in,ps_in = let spkgen = S.SGPoisson(ext_rate)
  sker = S.SKExp(myτe_ker)
  nt = S.NTInputConductance(spkgen,sker,v_rev_e)
  ps = S.PSInputPoissonConductance(nt,je_in,Ne)
  (nt,ps)
end

## Connek !
pop_e = S.Population(ps_e,(conn_e_e,ps_e),(S.FakeConnection(),ps_in))
myntw = S.RecurrentNetwork(dt,pop_e)

## run for a very short time, and check stability
const Ttot = 0.5
const krec = 1
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)#;idx_save=collect(1:100))
rec_spikes = S.RecSpikes(ps_e,50.0,Ttot)

## Run
times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_spikes,rec_state_e])
S.reset!(ps_e)
S.reset!(conn_e_e)
# initial conditions
ps_e.state_now .= rand(Uniform(v_reset_e,-0.1),Ne)

@time begin
  @showprogress 1.0 "network simulation " for (k,t) in enumerate(times)
    rec_spikes(t,k,myntw)
    rec_state_e(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
end


#S.add_fake_spikes!(vth_e,rec_state_e,rec_spikes)
## plot a raster
therast1 = S.raster_png(0.001,rec_spikes;spike_height=3,Nneurons=Ne,Ttot=Ttot);
save("/tmp/tmp.png",therast1)

## plot the state for one neuron

_ = let neu = 1,
  plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[neu,:];linewidth=2,leg=false,
    xlims=(0,1))
end

spkdic=S.get_spiketimes_dictionary(rec_spikes)

# initial conditions...

# measure all spikes




# now, let's add global inhibitory stabilization (aka cheating fake stabilization)



# wow, so dynamical stability!  Much asynchronous irregular! 


