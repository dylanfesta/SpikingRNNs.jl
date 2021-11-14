push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools,Test
using ProgressMeter


using Random; Random.seed!(0)
##
const Ntot = 1000
const Ttot = 5.0
const mydensity = 0.3
const dt = 0.1E-3
const rates = rand(Uniform(20.,50.),Ntot)

# dummy neurons

nt_all = let sker = S.SKExp(Inf)
  spkgen = S.SGPoissonMulti(rates)
  S.NTInputConductance(spkgen,sker,-Inf)
end
ps_all = S.PSInputConductance(nt_all,Ntot)

## initial weights
weights_start = let ρstart = mydensity
  ret = S.sparse_constant_wmat(Ntot,Ntot,ρstart,1.0;no_autapses=true)
  ret
end


# test 3 :  weight-dependent structural plasticity,
# half synapses are killed in several realizations
# show weight distribution before and after.

weight_gen_distr = Uniform(5.0,30.0) 
weights_start = let ρstart = mydensity
  ret = S.sparse_constant_wmat(Ntot,Ntot,ρstart,1.0;no_autapses=true)
  retnz = nonzeros(ret)
  rand!(weight_gen_distr,retnz)
  ret
end

## plasticity rule
plast_struct = let  νdeath = log(10)/(Ttot),
  temperature = 100.0,
  Δt = 30E-3,
  ρ = 0.0,
  syngen = S.SynapticGenerationConstant(2.0)
  srt_type = S.StructuralPlasticityWeightDependent(; 
    connection_density=ρ,
    death_rate = νdeath, w_temperature = temperature )
  S.PlasticityStructural(srt_type,syngen,Δt)
end

##

conn_all=S.ConnectionPlasticityTest(copy(weights_start),plast_struct)
pop_all=S.Population(ps_all,(conn_all,ps_all))
myntw = S.RecurrentNetwork(dt,pop_all)

## Run
wstart = copy(weights_start)

function _one_run()
  times = (0:myntw.dt:Ttot)
  # reset weights
  copy!(conn_all.weights,weights_start)
  S.reset!(ps_all)
  S.reset!(conn_all)
  for t in times
    S.dynamics_step!(t,myntw)
  end
  wend = conn_all.weights
  return copy(nonzeros(wend))
end

wend_all = []
nsampl = 3
@showprogress for k in 1:nsampl
  push!(wend_all, _one_run())
end

wend_all = vcat(wend_all...)


histogram(nonzeros(wstart))
_ = let plt=plot()
  bins = range(0,35;length=80)
  h1 = normalize(fit(Histogram,nonzeros(wstart),bins))
  h2 = normalize(fit(Histogram,wend_all,bins))
  plot(h1;opacity = 0.5,label="w start")
  plot!(h2;opacity = 0.5,label="w end")
end
