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

triplets_plasticity = let (n_post,n_pre) = size(wmat_start),
  τplus = 17E-3 # 16.8 
  τminus = 34E-3 # 33.7
  τx = 100E-3 # 101
  τy = 120E-3 # 125
  plast_eps = 1E-3
  A2plus = 7.5E-7*plast_eps
  A3plus = 9.3*plast_eps
  A2minus = 7.0*plast_eps
  A3minus = 2.3E-1*plast_eps
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
tw,weights_all = S.rebuild_weights(rec_weights)
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
  x = rec_weights.times
  n = size(weights_all[1],1)
  pre = 1
  for post in 1:n
    if post != pre
      y = getindex.(weights_all,post,pre)
      plot!(plt,x,y;linewidth=2,label="$(post) <- $(pre)")
    end
  end
  plot(plt)
end

##

##############################
# Let's make a function to test it 

function test_plasticity(lowrate::R,highrate::R,
    myplasticity::S.PlasticityRule;(Ttot::R)=50.0,(dt::R)=0.1E-3) where R
  nneus = 2
  nt,ps = let spkgen = S.SGPoissonMulti([highrate,lowrate])
    sker = S.SKExp(-1E6) # this will never be used
    vrev = -3000.0 # same
    nt = S.NTInputConductance(spkgen,sker,vrev)
    ps = S.PSInputConductance(nt,nneus)
    (nt,ps)
  end
  wstart = let w = 100.0
    wm = fill(w,(nneus,nneus))
    wm[diagind(wm)] .= 0.0
    sparse(wm)
  end
  conn = S.ConnectionPlasticityTest(wstart,myplasticity)
  pop = S.Population(ps,(conn,ps))
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
  for (k,t) in enumerate(times)
    rec_spikes(t,k,ntw)
    rec_weights(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
  # FROM HIGH TO LOW
  idx_low_high = findfirst(idx->Tuple(idx)==(2,1),rec_weights.idx_save)
  w_low_high = rec_weights.weights_now[idx_low_high,:]
  # FROM LOW TO HIGH
  idx_high_low = findfirst(idx->Tuple(idx)==(1,2),rec_weights.idx_save)
  w_high_low = rec_weights.weights_now[idx_high_low,:]
  _dostuff = function(v::Vector{R})
    vd=diff(v)
    filter!(!=(0.0),vd)
    sum(vd)/Ttot
  end
  return _dostuff.((w_low_high,w_high_low))
end
##
triplets_plasticity = let (n_post,n_pre) = size(wmat_start),
  τplus = 17E-3 # 16.8 
  τminus = 34E-3 # 33.7
  τx = 100E-3 # 101
  τy = 120E-3 # 125
  plast_eps = 1E-3
  A2plus = 7.5E-7*plast_eps
  A3plus = 9.3*plast_eps
  A2minus = 7.0*plast_eps
  A3minus = 2.3E-1*plast_eps
  S.PlasticityTriplets(τplus,τminus,τx,τy,A2plus,A3plus,
    A2minus,A3minus,n_post,n_pre)
end

booh = test_plasticity(10.0,80.0,triplets_plasticity)

##
nrats = 25
ratestest = range(0.5,60;length=nrats)
testret = fill(NaN,nrats,nrats)

for i in 1:nrats, j in i:nrats
  # j is high, i is low or equal
  @info "now processing $i,$j (out of $nrats)"
  (low_high,high_low) = test_plasticity(ratestest[i],
    ratestest[j], triplets_plasticity ; Ttot=25.0)
  testret[i,j] = low_high
  testret[j,i] = high_low
end
##

heatmap(ratestest,ratestest,testret;color=:vik)

function plus_minus_rescale(mat::Array{R}) where R
  down,up = extrema(mat)
  @assert (down < 0.0) && (up > 0.0)
  pluspart = mat ./ up
  minuspart = mat ./ abs(down)
  ret = similar(mat)
  for i in eachindex(mat)
    if mat[i] > 0
      ret[i] = pluspart[i]
    else
      ret[i] = minuspart[i]
    end
  end
  return ret
end

heatmap(ratestest,ratestest,testret;
  color=:seismic,ratio=1)

heatmap(ratestest,ratestest,plus_minus_rescale(testret);
  color=:seismic,ratio=1)



