# Stationary voltage density (and stationary rate) for a single LIF neuron
# without refractoriness!

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
Ttot = 30.

## start with LIF model

# LIF neuron parameters
myτ = 0.5
vth = 10.
v_r = -5.0
τrefr = 0.5 # refractoriness
τpcd = 0.2 # post synaptic current decay
myinput = 20.0
mynt = S.NTLIF(myτ,vth,v_r,τrefr,τpcd)
myps = S.PSLIF(mynt,1)

## one static input above threshold 
hw_in=20.0
in_state = S.PSSimpleInput(S.InputSimpleOffset(hw_in))
mypop = S.Population(myps,(S.FakeConnection(),),(in_state,))


## that's it, let's make the network
myntw = S.RecurrentNetwork(dt,(mypop,))

## Record spike number and internal potential
myrec1 = S.RecStateNow(myps,10,dt,Ttot)
myrec2 = S.RecCountSpikes(myps,dt)

## Run the simulation
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
## show internal membrane potential
_ = let tshow = (0.,5.)
  ts = myrec1.times
  y = myrec1.state_now[1,:]
  idx_keep = findall(t->tshow[1]<= t <= tshow[2],ts)
  plot(ts[idx_keep],y[idx_keep];leg=false,linewidth=2)
end
##

# now, use Fokker-Planck theory and Richardson method
# it is hardcoded for now... maybe later I will convert it to 
# a function
dv = 1E-3
v_lb = v_r - 0.1
ndv = round(Int64,(vth - v_lb)/dv)
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
  ps[k-1] = js[k-1] * mynt.τ / (myinput-Vks[k])
end

r0 = inv(dv*sum(ps))

##
_ = let plt = plot(;xlabel="membrane potential",ylabel="prob. density",
    leg=:topleft)
  histogram!(plt,myrec1.state_now[:];nbins=100,normalize=true,label="simulation")
  plot!(plt,Vks,r0.*ps;color=:white,linewidth=2,label="Fokker-Plank")
end

## Let's do an f-I curve with this

function f_lif_num(h_in;dt=1E-4,Ttot=10.0)
  in_state = S.PSSimpleInput(S.InputSimpleOffset(h_in))
  mypop = S.Population(myps,(S.FakeConnection(),),(in_state,))
  myntw = S.RecurrentNetwork(dt,(mypop,))
  rec = S.RecCountSpikes(myps,dt)
  ## Run!
  times = (0:myntw.dt:Ttot)
  # initial conditions
  myps.state_now[1] = v_r
  S.reset!(rec)
  for (k,t) in enumerate(times)
    S.dynamics_step!(t,myntw)
    rec(t,k,myntw)
  end
  r0 = S.get_mean_rates(rec,dt,Ttot)[1]
  return r0
end

function f_lif_semianalitic(h_in;dv=1E-3)
  v_lb = v_r - 0.1
  ndv = round(Int64,(vth - v_lb)/dv)
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
    ps[k-1] = js[k-1] * mynt.τ / (h_in-Vks[k])
  end
  return inv(dv*sum(ps))
end

##
_ = let Ttot = 50.0
  nsim = 40
  nfp = 200
  inputs_sim = range(vth+0.01,80.0;length=nsim)
  inputs_fp = range(vth+0.01,80.0;length=nfp)
  y1 = @showprogress map(in->f_lif_num(in;Ttot=Ttot),inputs_sim)
  y2 = @showprogress map(in->f_lif_semianalitic(in),inputs_fp)
  plt = plot(leg=:bottomright,xlabel="input current",ylabel="stationary rate")
  scatter!(plt,inputs_sim,y1;label="simulation")
  plot!(plt,inputs_fp,y2; label="Fokker-Planck",linewidth=2) 
end


#####################################
#####################################
## now a LIF neuron with a gaussian noisy input
# Time parameters
dt = 1E-4
Ttot = 500.
# LIF neuron parameters
myτ = 0.5
v_th = 10.
v_r = -5.0
τrefr = 0.0 # refractoriness
τpcd = 0.2 # post synaptic current decay
myinput = 7.
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
rec1_twup=50.0 # further away from initial conditions
myrec1 = S.RecStateNow(myps,10,dt,Ttot;t_warmup=rec1_twup)
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
_ = let tshow = (0.,8.)
  tshow = tshow .+ rec1_twup
  ts = myrec1.times
  y = myrec1.state_now[1,:]
  idx_keep = findall(t->tshow[1]<= t <= tshow[2],ts)
  plot(ts[idx_keep],y[idx_keep];leg=false,linewidth=2,xlabel="time",
    ylabel="membrane potential")
end

## now, use Fokker-Planck theory and Richardson method
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
  Bk = Gk==0 ? dv*Hk : Hk/Gk*(Ak-1)
  ps[k-1] = ps[k]*Ak + Bk
end

r0 = inv(dv*sum(ps))

_ = let plt = plot(;xlabel="membrane potential",ylabel="prob. density",
    leg=:topleft)
  histogram!(plt,myrec1.state_now[:];nbins=100,normalize=true,label="simulation")
  plot!(plt,Vks,r0.*ps;color=:white,linewidth=2,label="Fokker-Plank")
end