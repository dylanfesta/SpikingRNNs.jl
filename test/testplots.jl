# push!(LOAD_PATH, abspath(@__DIR__,".."))

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

dt = 0.1E-3
myτe = 20E-3 # seconds
myτi = 20E-3 # seconds
τrefr= 1E-3 # refractoriness
vth_e = 20.   # mV
vthexp = -52.0 # actual threshold for spike-generation
vth_i = vthexp
eifslope = 2.0
Cap = 300.0 #capacitance mF
v_rest_e = -60.0
v_rest_i = -60.0
v_rev_e = 0.0
v_rev_i = -75.0
v_leak_e = v_rest_e
v_leak_i = v_rest_i
v_reset_e = v_rest_e
v_reset_i = v_rest_i

# synaptic kernel
taueplus = 6E-3 #e synapse decay time
taueminus = 1E-3 #e synapse rise time

tauiplus = 2E-3 #i synapse decay time
tauiminus = 0.5E-3 #i synapse rise time

# input 
const Ne = 500
nt_in = let in_rate = 1E3
  sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoisson(in_rate)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
ps_in_e = S.PSInputPoissonConductanceExact(nt_in,1.0,Ne)

##

# TODO complete tests

genfun1(t::Real) = 10*rem(t,3.0) 
genfun2(t::Real) = ZZZZZZZZZZZZzz


sgentest = S.SGPoissonFExact(therate)

t_final = let t_current = 0.0
  for i in 1:1_000
    t_current=S._get_spiketime_update(t_current,sgentest,2)
  end
  t_current
end
ratenum = 1_000/t_final


##


sgentest = S.SGPoisson(10E3)

spiketimes = let ret=Float64[]
  t_current = 0.0
  for i in 1:10_000
    push!(ret,t_current)
    t_current=S._get_spiketime_update(t_current,sgentest,2)
  end
  ret
end

scatter(spiketimes[1:100],fill(1,100),marker=:vline)

length(spiketimes)/spiketimes[end]

## 
nt_e = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SpikeGenEIF(vthexp,eifslope)
  S.NTLIFConductance(sker,sgen,myτe,Cap,
   vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
end
ps_e = S.PSLIFConductance(nt_e,Ne)

nt_i = let sker = S.SKExpDiff(tauiplus,tauiminus)
  sgen = S.SpikeGenNone()
  S.NTLIFConductance(sker,sgen,myτi,Cap,
   vth_i,v_reset_i,v_rest_i,τrefr,v_rev_i)
end
ps_i = S.PSLIFConductance(nt_i,Ni)


nt_in_i = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoisson(in_rate_i)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
ps_in_i = S.PSInputPoissonConductance(nt_in_i,Jin_i,Ni)













##



# let's make a network with strong E-E connections that explodes

# define E population of conductance based EIF neurons
const dt = 0.1E-3
const Ne = 500

myτe = 20E-3 # seconds
τrefr= 5E-3 # refractoriness
vth_e = 20.   # mV
myτe_ker = 15E-3 # synaptic kernel 
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
const jee = 4.0
const pee = 0.33333
wmat = S.sparse_constant_wmat(Ne,Ne,pee,jee;no_autapses=true)
conn_e_e = S.ConnGeneralIF2(wmat)

## define external input of Poisson neurons
const ext_rate = 80.0
const je_in = 60.0
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
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot;idx_save=collect(1:100))
rec_spikes = S.RecSpikes(ps_e,1000.0,Ttot;idx_save=collect(1:100))

## Run
times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_spikes,rec_state_e])
S.reset!(ps_e)
S.reset!(conn_e_e)
# initial conditions
ps_e.state_now .= rand(Uniform(v_reset_e,-50),Ne)

@time begin
  @showprogress 1.0 "network simulation " for (k,t) in enumerate(times)
    rec_spikes(t,k,myntw)
    rec_state_e(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
end

S.add_fake_spikes!(vth_e,rec_state_e,rec_spikes)
## plot a raster
therast1 = S.raster_png(1E-3,rec_spikes;
  spike_height=3,Ttot=Ttot,Nneurons=length(rec_spikes.idx_save));
save("/tmp/tmp.png",therast1)

## plot the state for one neuron

_ = let neu = 1,
  plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[neu,:];linewidth=2,leg=false)
end

# explodes quite badly!

## now, let's add global inhibitory stabilization (aka cheating fake stabilization)
ps_istab = let v_rev_i = -70.0
  Aloc = 500.0
  Aglo = 1000.0
  τloc = 50E-3
  τglo = 5E-3
  S.PSConductanceInputInhibitionStabilization(v_rev_i,Aglo,Aloc,τglo,τloc,Ne)
end

## connect with stabilizer
pop_e_stab = S.Population(ps_e,
  (conn_e_e,ps_e),
  (S.FakeConnection(),ps_in),
  (S.FakeConnection(),ps_istab))

myntw_stab = S.RecurrentNetwork(dt,pop_e_stab)

const Ttot = 5.0
const krec = 1
rec_state_e_stab = S.RecStateNow(ps_e,krec,dt,Ttot;idx_save=collect(1:100),
  t_warmup=4.0)
rec_spikes_stab = S.RecSpikes(ps_e,1000.0,Ttot;idx_save=collect(1:100),
  t_warmup=4.0)

## Run
times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_spikes_stab,rec_state_e_stab])
S.reset!.([ps_e,ps_in,ps_istab])
S.reset!(conn_e_e)
# initial conditions
ps_e.state_now .= rand(Uniform(v_reset_e,-60),Ne)

@time begin
  @showprogress 1.0 "network simulation " for (k,t) in enumerate(times)
    rec_spikes_stab(t,k,myntw_stab)
    rec_state_e_stab(t,k,myntw_stab)
    S.dynamics_step!(t,myntw_stab)
  end
end

S.add_fake_spikes!(vth_e,rec_state_e_stab,rec_spikes_stab)
## plot a raster
therast2 = S.raster_png(1E-3,rec_spikes_stab;spike_height=3,
  Nneurons=100,Ttot=Ttot)
save("/tmp/tmp2.png",therast2)

_ = let neu = 1,
  plt=plot()
  plot!(plt,rec_state_e_stab.times,rec_state_e_stab.state_now[neu,:];linewidth=1.5,leg=false)
end

# wow, so dynamical stability!  Much asynchronous irregular! 


