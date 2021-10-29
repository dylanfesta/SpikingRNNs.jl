
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
const p_as = 0.2

const Δt_as = 0.2
const Δt_blank = 0.1
const t_as_delay = 0.3

const Ttot = 20.0

# input to all neurons
const lowrate = 1E3
# additional input to assembly neurons
const highrate = 5E3
##

as_idxs = map(_->findall(rand(Ne) .< p_as),1:Nas)

inputfun=S.pattern_functor(Δt_as,Ttot,lowrate,highrate,Ntot,
  as_idxs;t_pattern_delay=t_as_delay,Δt_pattern_blank=Δt_blank)
inputfun_upper = S.pattern_functor_upperlimit(lowrate,highrate,Ntot,as_idxs)  

## let's visualize it !
_  = let times = range(-1,Ttot+3;length=500)
  ys = hcat([inputfun.(t,1:Ne) for t in times]...)
  heatmap(times,1:Ntot,ys)
end

idxs_sort = S.order_by_pattern_idxs(as_idxs,Ne)

_  = let times = range(-1,Ttot+3;length=500)
  ys = hcat([inputfun.(t,1:Ne) for t in times]...)
  heatmap(times,1:Ntot,ys[idxs_sort,:])
end


##
# let's say kernel and v_reversal are same as E population 
# excitatory neurons are Auguste's
const taueplus = 6E-3
const taueminus = 1E-3
const eifslope = 2.0
const τe = 20E-3
const Cap = 300.0
const vthexp = -50.0
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


const w_in = 3.0
nt_in = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoissonFExact(inputfun,inputfun_upper)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
ps_in = S.PSInputPoissonConductanceExact(nt_in,w_in,Ntot)

## now define E to E connections
# with triplets plasticity rule!
const jee = 6.0 # this should scale as sqrt(Ne), or something like that
const pee = 0.2
wmat_start = S.sparse_constant_wmat(Ne,Ne,pee,jee;no_autapses=true)
# now the plasticity
# parameters from Augiste's code
triplets_plasticity = let τplus = 20E-3,
  τplus = 25E-3 # 16.8 
  τminus = 33E-3 # 33.7
  τx = 100E-3 # 101
  τy = 120E-3 # 125
  plast_eps = 5E-5
  A2plus = 0.075*plast_eps
  A3plus = 9.3*plast_eps
  A2minus = 7.0*plast_eps
  A3minus = 0.2*plast_eps
  (n_post,n_pre) = size(wmat_start)
  S.PlasticityTriplets(τplus,τminus,τx,τy,A2plus,A3plus,
    A2minus,A3minus,n_post,n_pre)
end

conn_e_e = S.ConnGeneralIF2(copy(wmat_start),triplets_plasticity)


## Now add stabilization to E neurons 

ps_istab = let v_rev_i = v_reset_e
  Aloc = 800.0
  Aglo = 2000.0
  τloc = 50E-3
  τglo = 5E-3
  S.PSConductanceInputInhibitionStabilization(v_rev_i,Aglo,Aloc,τglo,τloc,Ne)
end

##
pop_e = S.Population(ps_e,
  (S.FakeConnection(),ps_in),
  (S.FakeConnection(),ps_istab),
  (conn_e_e,ps_e))

## network 
const dt = 0.1E-3
ntw = S.RecurrentNetwork(dt,pop_e)

# record spiketimes and internal potential
krec = 1
n_e_rec = 3
t_wup = 0.0
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_spikes_e = S.RecSpikes(ps_e,500.0,Ttot;idx_save=collect(1:Ne),t_warmup=t_wup)

## Run

times = (0:ntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e])
S.reset!.([ps_e,ps_in,ps_istab])
# initial conditions
ps_e.state_now .= v_reset_e #rand(Uniform(v_reset_e,v_reset_e+0.3),Ne)

@time begin
  @showprogress 5.0 "network simulation " for (k,t) in enumerate(times)
    rec_state_e(t,k,ntw)
    rec_spikes_e(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
end

S.add_fake_spikes!(1.0vth_e,rec_state_e,rec_spikes_e)

## visualize activity

_ = let neu = 1,
  plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[neu,:];linewidth=2,leg=false)
end

##
idxs_sort = S.order_by_pattern_idxs(as_idxs,Ne)
_ = let Traster = Ttot
  tmp = S.raster_png(1E-3,rec_spikes_e;spike_height=3,Ttot=Traster,Nneurons=Ne,
   reorder=idxs_sort)
  save("/tmp/tmp.png",tmp)
end

##
_ = let (_times,binned) =S.binned_spikecount(0.1,rec_spikes_e)
  heatmap(_times,1:size(binned,1),binned[idxs_sort,:])
end

# order E neurons by assembly

## 
w_start = copy(wmat_start)
w_end = copy(conn_e_e.weights)
w_diff = w_end .- w_start

##
histogram(nonzeros(w_start))
histogram(nonzeros(w_end))
histogram(nonzeros(w_diff))

heatmap(Matrix(w_end)[idxs_sort,idxs_sort])
heatmap(Matrix(w_end)[idxs_sort,idxs_sort])

heatmap(Matrix(w_end))