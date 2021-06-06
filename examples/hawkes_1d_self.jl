
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

using FFTW

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

# numeric
@show nspikes/myspktimes[end]
# analytic from exponent
@show S.hawkes_exp_self_mean(myin,mywself,myβ)
# analytic given FFW of self-interaction kernel
# between eq 6 and eq 7 in Hawkes 1997
@show let mydt = 0.01,
  myτmax = 10.0,
  timescov = mydt:mydt:myτmax,
  gfou = fft( @. mywself*exp(-myβ*timescov)).*mydt
  inv(1-real(gfou[1]))*myin 
end;

##


# what about the covariance density?

mydt = 1.0
myτmax = 10.0

cov_num,binsc = S.covariance_self_numerical(myspktimes,myτmax,mydt,myspktimes[end])
cov_an = S.hawkes_exp_self_cov(binsc,myin,mywself,myβ)

plot(binsc,[cov_an cov_num];leg=false,linewidth=3)

##

# Hawkes 2D , but without interactions
# it's the 1D case repeated twice...

myβ = 0.8
mywself_a = 0.2
mywself_b = 0.1
mywmat = sparse([ 0.2 0. 
            0  0.1 ]) 
myin = [ 1.123,3.0]
tfake = NaN
p1 = S.PopulationHawkesExp(2,myβ)
ps1 = S.PSHawkes(p1)
conn1 = S.ConnectionHawkes(ps1,mywmat,ps1)
# rates to test
p1_in = S.PopInputStatic(ps1,myin)
myntw = S.RecurrentNetwork(tfake,(ps1,),(p1_in,),(conn1,) )

#
nspikes = 200_000
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

myspk1 = S.hawkes_get_spiketimes(1,my_act)
myspk2 = S.hawkes_get_spiketimes(2,my_act)
myTmax = max(myspk1[end],myspk2[end])+eps()
@info "Total duration $(round(myTmax;digits=1)) s"

##
# now mean rate, covariance, etc
@show S.hawkes_exp_self_mean(myin[1],mywmat[1,1],myβ)
@show length(myspk1)/myspk1[end]

@show S.hawkes_exp_self_mean(myin[2],mywmat[2,2],myβ)
@show length(myspk2)/myspk2[end]

## what about the covariance density?

mydt = 1.0
myτmax = 10.0

spikes_both=[myspk1,myspk2]

covtimes, cov_densities = 
  S.covariance_density_numerical(spikes_both,myτmax,mydt,myTmax)

cov_an1 = S.hawkes_exp_self_cov(covtimes,myin[1],mywmat[1,1],myβ)
cov_an2 = S.hawkes_exp_self_cov(covtimes,myin[2],mywmat[2,2],myβ)

plot(covtimes,[cov_an1 cov_densities[1][2]];leg=false,linewidth=3)
plot(covtimes,[cov_an2 cov_densities[3][2]];leg=false,linewidth=3)
# this is zero, because they are not connected
plot(covtimes,cov_densities[2][2];leg=false,linewidth=3)
##