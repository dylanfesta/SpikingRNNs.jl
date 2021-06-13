using Distributions: StatsBase
using LinearAlgebra: real
using Base: _maybetail

push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; plotlyjs() ; theme(:dark)
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


myβ = 1.33
mywself = 0.7
myin = 3.7
tfake = NaN
p1 = S.PopulationHawkesExp(1,myβ)
ps1 = S.PSHawkes(p1)
conn1 = S.ConnectionHawkes(ps1,onesparsemat(mywself),ps1)
# rates to test
p1_in = S.PopInputStatic(ps1,[myin,])
myntw = S.RecurrentNetwork(tfake,(ps1,),(p1_in,),(conn1,) )

#
nspikes = 200_000
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
numrate = nspikes/myspktimes[end]
@info "Total duration $(round(myspktimes[end];digits=1)) s"
@info "Rate : $(round(numrate;digits=2)) Hz"

##
# now mean rate, covariance, etc

# numeric
# analytic from exponent
@show S.hawkes_exp_self_mean(myin,mywself,myβ)
# analytic given FFW of self-interaction kernel
# between eq 6 and eq 7 in Hawkes 1997
@show let mydt = 0.0001,
  myτmax = 50.0,
  ts = S.get_times(mydt,myτmax)
  gfou = fft(S.interaction_kernel.(ts,p1)) .* (mywself*mydt)
  myin/(1-real(gfou[1])) 
end;
# or I can use the analytic transform in ω = 0. Samsies
ratefou = let gfou0 = mywself * S.fou_interaction_kernel.(0,p1)
  myin/(1-real(gfou0)) 
end

# this is to test analytic vs numeric fourier transform
# gfou = fft(mywself .* S.interaction_kernel.(mytaus,p1) ) .* mydt
# gfou2 = mywself .* S.fou_interaction_kernel.(myfreq,p1) |> ifftshift
##

# what about the covariance density
# First, compute it numerically for a reasonable time step
mydt = 0.2
myτmax = 30.0
mytaus = S.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = S.covariance_self_numerical(myspktimes,mydt,myτmax)
##
# now compute it analytically, at higher resolution

function four_high_res(dt::Real,Tmax::Real) # higher time resolution, longer time
  k=2
  myτmax = Tmax * k
  mytaus = S.get_times(dt,myτmax)
  nkeep = div(length(mytaus),k)
  myfreq = S.get_frequencies_centerzero(dt,myτmax)
  gfou = mywself .* S.fou_interaction_kernel.(myfreq,p1) |> ifftshift
  ffou = let r=ratefou
    covf(g) = r*(g+g'-g*g')/((1-g)*(1-g'))
    map(covf,gfou)
  end
  retf = real.(ifft(ffou)) ./ dt
  return mytaus[1:nkeep],retf[1:nkeep]
end

taush,covfou=four_high_res(0.1mydt,myτmax)

myplt = plot(mytaus[2:end], cov_num[2:end] ; linewidth=3, label="numerical" )
plot!(taush[1:end],covfou[1:end]; label="from Fourier",linewidth=3,linestyle=:dash)


##
# Hawkes 2D 

myβ = 2.1
mywmat = [ 0.31   0.5 
           0.8  0.15 ]
myin = [1.0,0.1]
tfake = NaN
p1 = S.PopulationHawkesExp(2,myβ)
ps1 = S.PSHawkes(p1)
conn1 = S.ConnectionHawkes(ps1,sparse(mywmat),ps1)
# rates to test
p1_in = S.PopInputStatic(ps1,myin)
myntw = S.RecurrentNetwork(tfake,(ps1,),(p1_in,),(conn1,) )

#
nspikes = 500_000
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
myTmax = my_act[end][2]+eps()
@info "Total duration $(round(myTmax;digits=1)) s"
@info "Rates are $(round.([length(myspk1)/myTmax, length(myspk2)/myTmax];digits=2))"

# analytic rate from fourier Eq between 6 and  7 in Hawkes
ratefou = let G0 =  mywmat .* S.fou_interaction_kernel.(0,p1)
  inv(I-G0)*myin |> real
end 

@info "With Fourier, rates are $(round.(ratefou;digits=2)) Hz"

##
# now mean rate, covariance, etc
# covariance numerical

mydt = 0.1
myτmax = 30.0
mytaus = S.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = S.covariance_density_numerical(myspikes_both,mydt,myτmax;verbose=true)

# plot numerical cov densities
plot(mytaus[2:end-1],cov_num[1,1,2:end-1] ; linewidth = 2)
plot!(mytaus[2:end-1],cov_num[1,2,2:end-1] ; linewidth = 2)
plot!(mytaus[2:end-1],cov_num[2,1,2:end-1] ; linewidth = 2)
plot!(mytaus[2:end-1],cov_num[2,2,2:end-1] ; linewidth = 2)

## now with fourier!
# Pick eq 12 from Hawkes
function four_high_res(dt::Real,Tmax::Real) 
  k1 = 2
  k2 = 0.2
  myτmax,mydt = Tmax * k1, dt*k2
  mytaus = S.get_times(mydt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = S.get_frequencies_centerzero(mydt,myτmax)
  G_omega = map(mywmat) do w
    ifftshift( w .* S.fou_interaction_kernel.(myfreq,p1))
  end
  D = Diagonal(ratefou)
  M = Array{ComplexF64}(undef,2,2,length(myfreq))
  Mt = similar(M,Float64)
  for i in eachindex(myfreq)
    G = getindex.(G_omega,i)
    # M[:,:,i] = (I-G)\(G*D+D*G'-G*D*G')/(I-G') 
    M[:,:,i] = (I-G)\D*(G+G'-G*G')/(I-G') 
  end
  for i in 1:2,j in 1:2
    Mt[i,j,:] = real.(ifft(M[i,j,:])) ./ mydt
  end
  return mytaus[1:nkeep],Mt[:,:,1:nkeep]
end


# Pick eq 13 from Hawkes
# does not really work :-/ 
function four_high_res2(dt::Real,Tmax::Real) 
  k=2
  myτmax = Tmax * k
  mytaus = S.get_times(dt,myτmax)
  nkeep = div(length(mytaus),k)
  myfreq = S.get_frequencies_centerzero(dt,myτmax)
  G_omega = map(mywmat) do w
    ifftshift( w .* S.fou_interaction_kernel.(myfreq,p1))
  end
  D = diagm(0=>ratefou)
  M = Array{ComplexF64}(undef,2,2,length(myfreq))
  for i in eachindex(myfreq)
    G = getindex.(G_omega,i)
    M[:,:,i] = (I-G)\D/(I-G') 
  end
  Mt = similar(M,Float64)
  for i in 1:2,j in 1:2
    Mt[i,j,:] = real.(ifft(M[i,j,:])) ./ dt
  end
  return mytaus[1:nkeep],Mt[:,:,1:nkeep]
end


taush,Cfou=four_high_res(mydt,myτmax)

plot(mytaus[2:end],cov_num[1,1,2:end] ; linewidth = 3)
plot!(taush[2:end],Cfou[1,1,2:end]; linestyle=:dash, linewidth=3)

plot(mytaus[2:end],cov_num[2,2,2:end] ; linewidth = 3)
plot!(taush[2:end],Cfou[2,2,2:end]; linestyle=:dash, linewidth=3)

plot(mytaus[2:end],cov_num[1,2,2:end] ; linewidth = 3)
plot!(taush[3:end],Cfou[1,2,3:end]; linestyle=:dash, linewidth=3)

plot(mytaus[2:end],cov_num[2,1,2:end] ; linewidth = 3)
plot!(taush[3:end],Cfou[2,1,3:end]; linestyle=:dash, linewidth=3)
