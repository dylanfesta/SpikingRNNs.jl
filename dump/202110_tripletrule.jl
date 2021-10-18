#=
Replicating Fig 1A in Gjorgjieva et al 2011
See also examples/plastictyFFtest.jl
=#


push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using ProgressMeter
using BenchmarkTools
function oneDSparse(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

##
dt = 0.1E-3
Ttot = 60.0 
function ps_trains(rate::R,Δt_ro::R,Ttot::R;
    tstart::R = 0.05,popτ::R=1E6) where R
  post = collect(range(tstart,Ttot; step=inv(rate)))
  pre = post .- Δt_ro
  return S.PSFixedSpiketrain([pre,post],popτ)
end
function ps_trains(rate::R,Δt_ro::R,Δt_oo::R,Ttot::R;
    tstart::R = 0.05,popτ::R=1E6) where R
  _post1 = collect(range(tstart,Ttot-2Δt_oo; step=inv(rate)))
  post = sort(vcat(_post1,_post1.+Δt_oo))
  pre = _post1 .- Δt_ro
  return S.PSFixedSpiketrain([pre,post],popτ)
end
## generic connection pre to post (2<-1)
wstart = sparse([ 0.0 0.0 ; 1000.0 0.0])

function get_weight_change(Δt_ro::R,plasticity::S.PlasticityRule;
   Δt_oo::Union{R,Nothing}=nothing,Ttot::R=80.0,dt::R=0.1E-3,
   rate::R=10.0,tstart::R=0.05,wstart::R=1E3) where R
  ps_all = if isnothing(Δt_oo)
      ps_trains(rate,Δt_ro,Ttot;tstart=tstart)
    else 
      ps_trains(rate,Δt_ro,Δt_oo,Ttot;tstart=tstart)
    end
  conn_all = S.ConnectionPlasticityTest(sparse([0.0 0.0 ; wstart 0.0]),
     plasticity)
  pop_all = S.Population(ps_all,(conn_all,ps_all))
  myntw = S.RecurrentNetwork(dt,pop_all)
  rec_weights = S.RecWeights(conn_all,krec,dt,Ttot)
  times = (0:myntw.dt:Ttot)
  S.reset!(rec_weights)
  S.reset!(ps_all)
  S.reset!(conn_all)
  for (k,t) in enumerate(times)
    rec_weights(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
  nsessions = length(ps_all.neurontype.trains[1])
  _wdiff = diff(rec_weights.weights_now[:])
  return sum(_wdiff)/nsessions
end

myplasticity = let τplus = 20E-3,
  τminus = 20E-3
  τx = 20E-3
  τy = 20E-3
  A2plus = 0.1
  A3plus = 0.1
  A2minus = 0.2
  A3minus = 0.2
  (n_post,n_pre) = size(wstart) 
  S.PlasticityTriplets(τplus,τminus,τx,τy,A2plus,A3plus,
    A2minus,A3minus,n_post,n_pre)
end


## negative part without second post spike
Δt_ros_minus = range(-80E-3,-1E-3;length=60)
wchange_minus =@showprogress map(
    Δt_ro-> get_weight_change(Δt_ro,myplasticity;rate=1.0),Δt_ros_minus)
plot(Δt_ros_minus,wchange_minus; marker=:circle)

## positive part with different post spikes

Δt_ros_plus = range(1E-3,80E-3;length=60)
wchange_plus1 = let Δt_oo = 100E-3
   @showprogress map(
    Δt_ro-> get_weight_change(Δt_ro,myplasticity;rate=1.0,Δt_oo=Δt_oo),Δt_ros_plus)
end
wchange_plus2 = let Δt_oo = 20E-3
   @showprogress map(
    Δt_ro-> get_weight_change(Δt_ro,myplasticity;rate=1.0,Δt_oo=Δt_oo),Δt_ros_plus)
end
wchange_plus3 = let Δt_oo = 10E-3
   @showprogress map(
    Δt_ro-> get_weight_change(Δt_ro,myplasticity;rate=1.0,Δt_oo=Δt_oo),Δt_ros_plus)
end

plt=plot(Δt_ros_plus,[wchange_plus1 wchange_plus2 wchange_plus3];
   marker=:circle,label=["100ms" "20ms" "10ms"])
plot!(plt, Δt_ros_minus,wchange_minus; marker=:circle,label="")

##

# to visualize a triplet
#=
_ = let _xlims = (0.,0.3)
  _ylims = (0.0 , 0.3)
  plt=plot(leg=false)
  spkdict = S.get_spiketimes_dictionary(rec_spikes)
  yvals = [0.1,0.2]
  for (k,v) in pairs(spkdict)
    scatter!(plt,_->yvals[k],v;marker=:vline,
      color=:white,markersize=11,
      ylims=_ylims,xlims=_xlims)
  end
  plt
end
=#