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

# kernel and its transform

exp_kernel(t,β,w) = w*exp(-β*t)
fouexp_kernel(ω,β,w) = w*inv(β - im*(2π)*ω)

function get_freqs(dt,T)
  dω = inv(2T)
  ωmax = inv(2dt)
  ret = collect(-ωmax:dω:(ωmax-dω))
  (val,idx) = findmin(abs.(ret))
  if abs(val)<1E-10
    ret[idx] = 1E-8
  end
  return ret
end

get_freqs(0.1,1.0)

##
# isolated, self-interacting , exp processes 
# theory Vs numerical

myβ = 0.8
mywself = 0.4
myweights = onesparsemat(mywself)
myin = 3.123
tfake = NaN
p1 = S.PopulationHawkesExp(1,myβ)
ps1 = S.PSHawkes(p1)
conn1 = S.ConnectionHawkes(ps1,myweights,ps1)
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

@show ratean = S.hawkes_exp_self_mean(myin,mywself,myβ)
@show ratenum = nspikes/myspktimes[end]

# analytic given FFW of self-interaction kernel
# between eq 6 and eq 7 in Hawkes 1997
@show let mydt = 0.01,
  myτmax = 10.0,
  timescov = mydt:mydt:myτmax,
  gfou = fft( @. mywself*exp(-myβ*timescov)).*mydt
  inv(1-real(gfou[1]))*myin 
end;


##
mydt = 0.1
myτmax = 40.0
timescov = 0:mydt:(myτmax-mydt)
cov_num,binsc = S.covariance_self_numerical(myspktimes,myτmax,mydt,myspktimes[end])
cov_an = S.hawkes_exp_self_cov(binsc,myin,mywself,myβ)

plot(binsc,[cov_num cov_an];leg=false,linewidth=3)
##

myfreq = get_freqs(mydt,myτmax)
gfou = fouexp_kernel.(myfreq,myβ,mywself)

ratefou = let fou0 = fftshift(gfou)[1]
  inv(1-real(fou0))*myin 
end


plot(myfreq,real.(gfou))
plot(myfreq,imag.(gfou))

plot(fftshift(real.(gfou)))
plot(fftshift(imag.(gfou)))

fftshift(gfou)

ffou = ratefou ./ ( norm.(1 .- fftshift(gfou) ).^2)*inv(mydt)

ifou = ifft(ffou)
plot(real.(ifou[2:end]))

_ = let a = 1 
  plot(timescov,real.(ifou[2:401]);lab="four")
  plot!(timescov ,cov_an ; lab="an")
end
