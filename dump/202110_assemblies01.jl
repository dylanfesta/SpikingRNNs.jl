
push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using ProgressMeter
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using Distributions
using Random ; Random.seed!(0)
using FileIO
##

const Ne = 500
const Ni = 0
const Ntot = Ne+Ni

const Nas = 5
const p_as = 0.05

const Δt_as = 1.0
const Ttot = 10.0

# input to all neurons
const lowrate = 5.0
# additional input to assembly neurons
const highrate = 50.0
##

wmat_input,pattidx = S.wmat_train_assemblies_protocol(Ne,Nas,p_as;scal=50.0)

patt_t,_=S._patterns_train_uniform(Nas,Δt_as,Ttot)

as_idx_pre = [p.neupre for p in  pattidx]
as_idx_post = [p.neupost for p in  pattidx]


inputfun=S.pattern_functor(Δt_as,Ttot,0.0,highrate,
  as_idx_pre;t_pattern_delay=5.0)


## let's visualize it !
_  = let times = range(-5,Ttot+10;length=500)
  npre = size(wmat_input,2)
  ys = hcat(inputfun.(times)...)
  heatmap(times,1:npre,ys)
end

##
# let's say kernel and v_reversal are same as E population 
# excitatory neurons are Auguste's
const taueplus = 6E-3
const taueminus = 1E-3
const eifslope = 2.0
const τe = 20E-3
const Cap = 300.0
const vthexp = -52.0
const v_rev_e = 0.0
const vth_e = 20.0
const v_reset_e = -60.0
const v_rest_e = -60.0
const τrefr  = 10E-3
nt_e = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SpikeGenEIF(vthexp,eifslope)
  S.NTLIFConductance(sker,sgen,τe,Cap,
   vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
end
ps_e = S.PSLIFConductance(nt_e,Ne)

nt_in = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoissonMultiF(inputfun)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
const n_input = size(wmat_input,2)
@assert n_input == length(inputfun(0.0))
ps_in = S.PSInputConductance(nt_in,n_input)

# now the constant input
nt_fix_in = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoisson(lowrate)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
ps_fix_in = S.PSInputConductance(nt_fix_in,Ne)
# and the connection for it
conn_e_fix_in = let wval = 50.0
  wmat = sparse(wval.*Matrix{Float64}(I,Ne,Ne))
  S.ConnGeneralIF2(wmat)
end


## Connection
conn_e_in = S.ConnGeneralIF2(wmat_input)


##

pop_e = S.Population(ps_e,(conn_e_in,ps_in),(conn_e_fix_in,ps_fix_in))
pop_in = S.Population(ps_in,(S.FakeConnection(),ps_in))
pop_fix_in = S.Population(ps_fix_in,(S.FakeConnection(),ps_fix_in))

## network 
global const dt = 0.1E-3
ntw = S.RecurrentNetwork(dt,pop_in,pop_fix_in,pop_e)

# record spiketimes and internal potential
krec = 20
n_e_rec = Ne
t_wup = 0.0
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_spikes_e = S.RecSpikes(ps_e,40.0,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_spikes_in = S.RecSpikes(ps_in,50.0,Ttot;t_warmup=t_wup)

## Run

times = (0:ntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e])
S.reset!.([rec_spikes_in])
S.reset!.([ps_e,ps_in])
S.reset!(conn_e_in)
# initial conditions
ps_e.state_now .= v_reset_e #rand(Uniform(v_reset_e,v_reset_e+0.3),Ne)

@time begin
  @showprogress 5.0 "network simulation " for (k,t) in enumerate(times)
    rec_state_e(t,k,ntw)
    rec_spikes_e(t,k,ntw)
    rec_spikes_in(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
end

S.add_fake_spikes!(1.0vth_e,rec_state_e,rec_spikes_e)

## visualize input neurons first

tmp = S.raster_png(0.01,rec_spikes_in;spike_height=3,Ttot=Ttot);
save("/tmp/tmp.png",tmp)

##
_ = let (_times,binned) =S.binned_spikecount(0.1,rec_spikes_in)
  heatmap(_times,1:size(binned,1),binned)
end

# order E neurons by assembly

idxs_sort = S.order_by_pattern_idxs(as_idx_post,Ne)
_ = let (_times,binned) =S.binned_spikecount(0.1,rec_spikes_e)
  heatmap(_times,1:size(binned,1),binned[idxs_sort,:])
end


tmp = S.raster_png(0.01,rec_spikes_e;Nneurons=Ne,spike_height=3,Ttot=Ttot,
  reorder=idxs_sort);
save("/tmp/tmp.png",tmp)

tmp = S.raster_png(0.01,rec_spikes_e;spike_height=3,Ttot=Ttot);


##
_ = let plt=plot()
  spkt,spkneu = S.get_spiketimes_spikeneurons(rec_spikes_in)
  for i in unique(spkneu)
    _idx = findall(==(i),spkneu)
    scatter!(plt,spkt[_idx],spkneu[_idx];marker=:vline,leg=false)
  end
  plt
end




