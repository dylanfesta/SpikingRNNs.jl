
push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using FFTW
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
S.hawkes_initialize!(myntw)

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
  ts = S.get_times(mydt,myτmax)
  gfou = fft( @. S.interaction_kernel.(ts,p1,mywself)).*mydt
  inv(1-real(gfou[1]))*myin 
end;

##

# what about the covariance density?

mydt = 0.1
myτmax = 10.0
mytaus = S.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = S.covariance_self_numerical(myspktimes,mydt,myτmax)
cov_an = S.hawkes_exp_self_cov(mytaus,myin,mywself,myβ)

myplt = plot(mytaus[2:end-1], cov_num[2:end-1] ; linewidth=3, label="numerical" )
plot!(myplt, mytaus, cov_an ;linewidth=3, label="full analytic")

## let's add to the plot the result with Fourier transform
myfreq = S.get_frequencies(mydt,myτmax)
gfou = S.fou_interaction_kernel.(myfreq,p1,mywself)
ratefou = let fou0 = fftshift(gfou)[1]
  inv(1-real(fou0))*myin 
end

ffou = ratefou ./ ( norm.(1 .- fftshift(gfou) ).^2)*inv(mydt)
ifou = ifft(ffou)


@info "zero lag covariance: numerical $(cov_num[1]) , with Fourier $(real(ifou[1]))"

plot!(myplt,mytaus[2:end], real.(ifou[2:ntaus]);
  label="from Fourier",linewidth=3,linestyle=:dash)


##
# Hawkes 2D , but without interactions

myβ = 0.5
mywmat = sparse([ 0.3   0.03 
                  1.3  0.2 ]) 
myin = [ 1.12,0.1]
tfake = NaN
p1 = S.PopulationHawkesExp(2,myβ)
ps1 = S.PSHawkes(p1)
conn1 = S.ConnectionHawkes(ps1,mywmat,ps1)
# rates to test
p1_in = S.PopInputStatic(ps1,myin)
myntw = S.RecurrentNetwork(tfake,(ps1,),(p1_in,),(conn1,) )

#
nspikes = 400_000
# initialize
S.hawkes_initialize!(myntw)

my_act = Vector{Tuple{Int64,Float64}}(undef,nspikes)
my_state1 = Vector{Float64}(undef,nspikes)
my_state2 = similar(my_state1)
for k in 1:nspikes
  S.dynamics_step!(tfake,myntw)
  idx_fire = findfirst(ps1.isfiring)
  t_now = ps1.time_now[1]
  my_act[k] = (idx_fire,t_now)
  my_state1[k] = ps1.state_now[1]
  my_state2[k] = ps1.state_now[2]
end

myspk1 = S.hawkes_get_spiketimes(1,my_act)
myspk2 = S.hawkes_get_spiketimes(2,my_act)
myspikes_both = [myspk1,myspk2]
myTmax = max(myspk1[end],myspk2[end])+eps()
@info "Total duration $(round(myTmax;digits=1)) s"
@info "Rates are $(length(myspk1)/myTmax) and $(length(myspk2)/myTmax)"

##
# now mean rate, covariance, etc

# covariance numerical

mydt = 0.4
myτmax = 80.0
mytaus = S.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = S.covariance_density_numerical(myspikes_both,mydt,myτmax;verbose=true)
cov_self1 = S.covariance_self_numerical(myspk1,mydt,myτmax)

##

# numerical cov densities
plot(mytaus[2:end-1],cov_num[1][2][2:end-1] ; linewidth = 2)
plot!(mytaus[2:end-1],cov_num[2][2][2:end-1] ; linewidth = 2)
plot!(mytaus[2:end-1],cov_num[3][2][2:end-1] ; linewidth = 2)

## now with fourier!
# G elements first

myfreq = S.get_frequencies(mydt,myτmax)
gfou_all  = map(convert(Matrix,mywmat)) do w
  S.fou_interaction_kernel.(myfreq,p1,w)
end
gfou_all_shift = map(fftshift,gfou_all)

ratefou = let gfou0 = getindex.(gfou_all_shift,1)
  real.(inv(I-gfou0)*myin) 
end

@info """
 Measured rates:  $(length(myspk1)/myTmax) and $(length(myspk2)/myTmax)
 Rates with Fourier : $(ratefou[1]) $(ratefou[2])
"""

# now the main one...

D = diagm(0=>ratefou)

Cfou = Array{eltype(gfou_all[1,1])}(undef,2,2,length(myfreq))

for k in 1:length(myfreq) 
  Gfm =  getindex.(gfou_all_shift,k)
  # the ' operator corresponds to transpose and -ω 
  Cfou[:,:,k] = ( inv((I-Gfm)) * D * inv((I-Gfm')) ) .* inv(mydt)
end

# C analytic
C_ana = mapslices(v-> real.(ifft(v)) ,Cfou;dims=3)


@info """
zero lag covariance (1,1): numerical  $(cov_num[1][2][1]) , with Fourier $(C_ana[1,1,1])
zero lag covariance (1,2): numerical  $(cov_num[2][2][1]) , with Fourier $(C_ana[1,2,1])
zero lag covariance (2,2): numerical  $(cov_num[3][2][1]) , with Fourier $(C_ana[2,2,1])
zero lag covariance (2,2): numerical  ?  , with Fourier $(C_ana[2,1,1])
"""


# I am not sure at all about the correction term

myplot = plot(mytaus[2:end],cov_num[1][2][2:end] ; linewidth = 2)
plot!(myplot,mytaus[2:end],C_ana[1,1,2:ntaus]  ; linewidth = 2)


myplot = plot(mytaus[2:end],cov_num[2][2][2:end] ; linewidth = 2)
plot!(myplot,mytaus[2:end],C_ana[2,1,2:ntaus] .- mydt^2*sqrt(ratefou[1]*ratefou[2]) ; linewidth = 2)

myplot = plot(mytaus[2:end],cov_num[3][2][2:end] ; linewidth = 2)
plot!(myplot,mytaus[2:end],C_ana[2,2,2:ntaus]  ; linewidth = 2)