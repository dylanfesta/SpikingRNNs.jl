# one LIF neuron with constant input

push!(LOAD_PATH, abspath(@__DIR__,".."))
using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using ProgressMeter

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

## Time parameters
dt = 1E-4
Ttot = 500.
# LIF neuron parameters
myτ = 0.5
v_th = 10.
v_r = -5.0
τrefr = 0.5 # refractoriness
τpcd = 0.2 # post synaptic current decay
myinput = 9.
myσ = 2.0
mynt = S.NTLIF(myτ,v_th,v_r,τrefr,τpcd)
myps = S.PSLIF(mynt,1)

## inputs here
# one static input
in_state = S.PSSimpleInput(S.InputSimpleOffset(myinput))
# ... plus noise
in_state_noise = S.PSSimpleInput(S.InputIndependentNormal(myσ))
##
mypop = S.Population(myps,(S.FakeConnection(),S.FakeConnection()),
    (in_state,in_state_noise))


## that's it, let's make the network
myntw = S.RecurrentNetwork(dt,(mypop,))

## Recorder
myrec1 = S.RecStateNow(myps,10,dt,Ttot;t_warmup=50.)
myrec2 = S.RecCountSpikes(myps,dt)

## Run!
times = (0:myntw.dt:Ttot)
nt = length(times)
# initial conditions
myps.state_now[1] = v_r

S.reset!.([myrec1,myrec2])

for (k,t) in enumerate(times)
  S.dynamics_step!(t,myntw)
  myrec1(t,k,myntw)
  myrec2(t,k,myntw)
end
r0 = S.get_mean_rates(myrec2,dt,Ttot)[1]
@info mean_and_std(myrec1.state_now[:])
##

# now, use Fokker-Planck theory and Richardson method
# it is hardcoded for now... maybe later I will convert it to 
# a function
dv = 1E-3
v_lb = v_r - 2.0
ndv = round(Int64,(v_th - v_lb)/dv)
k_re = round(Int64,(v_r - v_lb)/dv)
ks = collect(0:ndv) 
Vks = @. v_lb + ks * dv 
js = fill(NaN,ndv+1)
ps = fill(NaN,ndv+1)

ksrev = collect(ndv+1:-1:2)

js[ndv+1]=1.0
ps[ndv+1]=0.0

for k in ksrev
  krond = k==k_re ? 1.0 : 0.0
  js[k-1] = js[k] - krond
  Gk = (Vks[k]-myinput)/(myσ^2)
  Hk = myτ*js[k]/(myσ^2)
  Ak = exp(dv*Gk)
  Bk = Gk==0 ? dv*Hk :  Hk/Gk*(Ak-1)
  ps[k-1] = ps[k]*Ak + Bk
end

r0 = inv(dv*sum(ps))

##
histogram(myrec1.state_now[:];nbins=100,normalize=true,
  label="simulation")
plot!(Vks,r0.*ps;color=:white,linewidth=2,leg=:topleft,
  label="Fokker-Planck")

## repeat for EIF neuron  

dt = 1E-4
Ttot = 150.
# EIF neuron parameters
# τ::Float64 # time constant (membrane capacitance)
# g_l::Float64 # conductance leak
# v_expt::Float64 # exp term threshold
# steep_exp::Float64 # steepness of exponential term
# v_threshold::Float64 # spiking threshold 
# v_reset::Float64 # reset after spike
# v_leak::Float64 # reversal potential for leak term
# τ_refractory::Float64 # refractory time

τ = 0.5
g_l = 1.0
v_expt=10.0
steep_exp = 1.0
v_th = 20.
v_r = -5.0
τrefr = 0.0
v_leak = 0.0

myinput = 7.0
myσ = 3.0

mynt = S.NTEIF(τ,g_l,v_expt,steep_exp,v_th,v_r,v_leak,τrefr)
myps = S.PSEIF(mynt,1)

## inputs here
# one static input
in_state = S.PSSimpleInput(S.InputSimpleOffset(myinput))
# ... plus noise
in_state_noise = S.PSSimpleInput(S.InputIndependentNormal(myσ))
##
mypop = S.Population(myps,(S.FakeConnection(),S.FakeConnection()),
    (in_state,in_state_noise))


## that's it, let's make the network
myntw = S.RecurrentNetwork(dt,(mypop,))

## Recorder
myrec1 = S.RecStateNow(myps,10,dt,Ttot;t_warmup=50.)
myrec2 = S.RecCountSpikes(myps,dt)

## Run!
times = (0:myntw.dt:Ttot)
nt = length(times)
# initial conditions
myps.state_now[1] = v_r

S.reset!.([myrec1,myrec2])

for (k,t) in enumerate(times)
  S.dynamics_step!(t,myntw)
  myrec1(t,k,myntw)
  myrec2(t,k,myntw)
end
r0_sim = S.get_mean_rates(myrec2,dt,Ttot)[1]
@info mean_and_std(myrec1.state_now[:])
##

# now, use Fokker-Planck theory and Richardson method
# it is hardcoded for now... maybe later I will convert it to 
# a function
dv = 1E-3
v_lb = v_r - 2.0

psifun(V) = steep_exp * exp((V-v_expt)/steep_exp)

ndv = round(Int64,(v_th - v_lb)/dv)
k_re = round(Int64,(v_r - v_lb)/dv)
ks = collect(0:ndv) 
Vks = @. v_lb + ks * dv 
js = fill(NaN,ndv+1)
ps = fill(NaN,ndv+1)

ksrev = collect(ndv+1:-1:2)

js[ndv+1]=1.0
ps[ndv+1]=0.0

for k in ksrev
  krond = k==k_re ? 1.0 : 0.0
  js[k-1] = js[k] - krond
  Gk = (Vks[k]-myinput-psifun(Vks[k]))/(myσ^2)
  Hk = myτ*js[k]/(myσ^2)
  Ak = exp(dv*Gk)
  Bk = Gk==0 ? dv*Hk :  Hk/Gk*(Ak-1)
  ps[k-1] = ps[k]*Ak + Bk
end

r0_fp = inv(dv*sum(ps))

@info "rate sim $r0_sim , rate F-P $r0_fp"

##
histogram(myrec1.state_now[:];nbins=100,normed=true,
  label="simulation")
plot!(Vks,r0_fp.*ps;color=:white,linewidth=2,leg=:topleft,
  label="Fokker-Planck")
