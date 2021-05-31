
push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end


##
# isolated, self-interacting , exp processes 
# theory Vs numerical

myβ = 0.8
mywself = 0.2
myin = 1.123
tfake = NaN
p1 = S.PopulationHawkesExp(1,myβ)
ps1 = S.PSHawkes(p1)
conn1 = S.ConnectionHawkes(ps1,onesparsemat(mywself),ps1)
# rates to test
p1_in = S.PopInputStatic(ps1,[myin,])
myntw = S.RecurrentNetwork(tfake,(ps1,),(p1_in,),(conn1,) )

#
nspikes = 100_000
# initialize
ps1.state_now .= 1E-2
S.send_signal!(tfake,p1_in)

my_act = Vector{Tuple{Int64,Float64}}(undef,nspikes)
my_state = Vector{Float64}(undef,nspikes)
for k in 1:nspikes
  S.dynamics_step!(tfake,myntw)
  idx_fire = findfirst(ps1.isfiring)
  t_now = ps1.time_now[1]
  my_act[k] = (idx_fire,t_now)
  my_state[k] = ps1.state_now[1]
end


myspktimes = getindex.(my_act,2)
@info "Total duration $(round(myspktimes[end];digits=1)) s"

##
# now mean rate, covariance, etc

S.hawkes_exp_self_mean(myin,mywself,myβ)

nspikes/myspktimes[end]

# what about the covariance density?

mydt = 1.0
myτmax = 10.0

cov_num,binsc = S.covariance_self_numerical(myspktimes,myτmax,mydt,myspktimes[end])
cov_an = S.hawkes_exp_self_cov(binsc,myin,mywself,myβ)

plot(binsc,[cov_an cov_num];leg=false,linewidth=3)

##