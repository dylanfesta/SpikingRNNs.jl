#=

This is just a simple test for the new input type

I excite a subgroup of E neurons for specific intervals.
=#

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using Test
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
vth_e = -20.0   # mV
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

const Ne = 200
nt_e = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SpikeGenEIF(vthexp,eifslope)
  S.NTLIFConductance(sker,sgen,myτe,Cap,
   vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
end
ps_e = S.PSLIFConductance(nt_e,Ne)

# step like function for half the population

function genfun(t::Real,i::Integer)
  min = 100.0
  max = 200.0
  period = 1.0
  if i>100
    return min
  else
    trem = rem(t,period)
    return trem < 0.5period ? min : max 
  end
end
function genfun_upper(t::Real,i::Integer)
  min = 100.0
  max = 300.0
  if i>100
    return min
  else
    return max
  end
end

nt_in = let 
  sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoissonFExact(genfun,genfun_upper)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
in_weight = 10.0
ps_in = S.PSInputPoissonConductanceExact(nt_in,in_weight,Ne)

##
pop_e = S.Population(ps_e,(S.InputDummyConnection(),ps_in))
ntw = S.RecurrentNetwork(dt,pop_e)

Ttot = 4.0
# record spiketimes and internal potential
krec = 1
n_e_rec = Ne
t_wup = 0.0
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_spikes_e = S.RecSpikes(ps_e,500.0,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)

## Run

times = (0:ntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e])
S.reset!.([ps_e,ps_in])
# initial conditions
ps_e.state_now .= v_reset_e

@time begin
  @showprogress 5.0 "network simulation " for (k,t) in enumerate(times)
    rec_state_e(t,k,ntw)
    rec_spikes_e(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
end

S.add_fake_spikes!(1.0vth_e,rec_state_e,rec_spikes_e)
myrast = S.raster_png(0.01,rec_spikes_e;Nneurons=Ne,Ttot=Ttot)
save("/tmp/rast.png",myrast)
