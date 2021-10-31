#=
Very similar to examples/plasticityFFtest.jl
Here I consider triplet rule, two high frequency neurons and two low 
frequency ones (all Poisson).  Desired outcome: weight increases between the 
high firing neurons, decreases in all other cases. 
=#
push!(LOAD_PATH, abspath(@__DIR__,".."))

using Test,ProgressMeter
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools

using Random ; Random.seed!(1)

##
const dt = 0.1E-3
const Ttot = 20.0 
const highrate = 80.0 
const lowrate = 5.0
const nneus = 4

nt,ps = let spkgen = S.SGPoissonMulti([highrate,highrate,lowrate,lowrate])
  # spkgen = S.SGPoissonMulti([highrate,highrate])
  sker = S.SKExp(-1E6) # this will never be used
  vrev = -3000.0 # same
  nt = S.NTInputConductance(spkgen,sker,vrev)
  ps = S.PSInputConductance(nt,nneus)
  (nt,ps)
end


# plasticity and initial weights

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
  (n_post,n_pre) = (nneus,nneus)
  S.PlasticityTriplets(τplus,τminus,τx,τy,A2plus,A3plus,
    A2minus,A3minus,n_post,n_pre)
end

wstart = let w = 100.0
  wm = fill(w,(nneus,nneus))
  wm[diagind(wm)] .= 0.0
  sparse(wm)
end
conn = S.ConnectionPlasticityTest(wstart,triplets_plasticity)

pop = S.Population(ps,(conn,ps))

# that's it, let's make the network
ntw = S.RecurrentNetwork(dt,pop)

## recorders : spikes 
krec = 1
rec_spikes = S.RecSpikes(ps,highrate,Ttot)
rec_weights = S.RecWeights(conn,krec,dt,Ttot)

## Run

times = (0.0:ntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([ps,rec_spikes,rec_weights,conn.plasticities[1]])
# initial conditions

@time @showprogress for (k,t) in enumerate(times)
  rec_spikes(t,k,ntw)
  rec_weights(t,k,ntw)
  S.dynamics_step!(t,ntw)
end

spk_out = S.get_spiketimes_dictionary(rec_spikes)
@show S.get_mean_rates(rec_spikes,dt,Ttot)
##
# check spikes
_ = let plt=plot(leg=false)
  for i in 1:nneus
   scatter!(plt, _->i, spk_out[i];marker=:vline)
  end
  plt
 end

##

_ = let plt=plot(legend=:topleft,xlabel="time",ylabel="weight")
  x=rec_weights.times
  for (i,idxs) in enumerate(rec_weights.idx_save) 
    if i<=3
      y = rec_weights.weights_now[i,:]
      plot!(plt,x,y;linewidth=2,label="$(idxs[1]) <- $(idxs[2])")
    end
  end
  plot(plt)
end

# ##
# wt1 = rec_weights.weights_now[1,:]
# wt2 = rec_weights.weights_now[2,:]
# ##
# findall(!=(0),diff(wt1))
# findall(!=(0),diff(wt2))

# ##
# d1 = filter!(!=(0),diff(rec_weights.weights_now[1,:]))
# d2 = filter!(!=(0),diff(rec_weights.weights_now[2,:]))

# _ = let lim = 20
#   plot(1:lim, [d1[2:lim+1] d2[1:lim]],marker=:circle)
# end

# mean(rec_weights.weights_now[1,:] .- 100.0)
# mean(rec_weights.weights_now[2,:] .- 100.0)

# plot( (rec_weights.weights_now[1,:] .- 100))
# plot!( 2.2 .* (rec_weights.weights_now[2,:] .- 100.0))

# histogram(diff(rec_weights.weights_now[2,:]))
##

# _ = let x=rec_weights.times
#  y = rec_weights.weights_now[:]
#  spkdict = S.get_spiketimes_dictionary(rec_spikes)
#  plt=plot(leg=false)
#  plot!(plt,x,y;linewidth=2,leg=false)
#  for (neu,spkt) in pairs(spkdict)
#   scatter!(twinx(plt),_->neu,spkt ; 
#     marker=:vline,color=:green,ylims=(0.5,3))
#  end
#  plt
# end
