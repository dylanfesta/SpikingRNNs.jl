```@meta
EditURL = "https://github.com/dylanfesta/SpikingRNNs.jl/blob/master/examples/if_stationary_voltage_density.jl"
```

Stationary voltage density (and stationary rate) for a single LIF neuron
without refractoriness!

````@example if_stationary_voltage_density
using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:default)
using SparseArrays
using SpikingRNNs; const global S = SpikingRNNs
using ProgressMeter

function onesparsemat(w::Real)
  return sparse(cat(w;dims=2))
end

# Time parameters
const dt = 1E-4
const Ttot = 30.

# start with LIF model
````

LIF neuron parameters

````@example if_stationary_voltage_density
const myτ = 0.5
const cap = myτ
const vth = 10.
const v_r = -5.0
const vleak = 0.0
const τrefr = 0.0 # refractoriness
const myinput = 20.0
const ps = S.PSIFNeuron(1,myτ,cap,vth,v_r,vleak,τrefr);

# one static input above threshold
const hw_in=20.0
const in_const = S.IFInputCurrentConstant(hw_in)
const conn_in = S.ConnectionIFInput([1.]);

const pop = S.Population(ps,(conn_in,in_const))


# that's it, let's make the network
const ntw = S.RecurrentNetwork(dt,pop)

# Record spike number and internal potential
myrec1 = S.RecStateNow(ps,10,dt,Ttot)
myrec2 = S.RecCountSpikes(ps,dt)

# Run the simulation
times = (0:ntw.dt:Ttot)
nt = length(times)
````

initial conditions

````@example if_stationary_voltage_density
ps.state_now[1] = v_r

S.reset!.([myrec1,myrec2])

for (k,t) in enumerate(times)
  S.dynamics_step!(t,ntw)
  myrec1(t,k,ntw)
  myrec2(t,k,ntw)
end
r0 = S.get_mean_rates(myrec2,dt,Ttot)[1]
# show internal membrane potential
_ = let tshow = (0.,5.)
  ts = myrec1.times
  y = myrec1.state_now[1,:]
  idx_keep = findall(t->tshow[1]<= t <= tshow[2],ts)
  plot(ts[idx_keep],y[idx_keep];leg=false,linewidth=2)
end

#
````

now, use Fokker-Planck theory and Richardson method
it is hardcoded for now... maybe later I will convert it to
a function

````@example if_stationary_voltage_density
const dv = 1E-3
const v_lb = v_r - 0.1
const ndv = round(Int64,(vth - v_lb)/dv)
const k_re = round(Int64,(v_r - v_lb)/dv)
const ks = collect(0:ndv)
const Vks = @. v_lb + ks * dv
const js = fill(NaN,ndv+1)
const pps = fill(NaN,ndv+1)

const ksrev = collect(ndv+1:-1:2)

js[ndv+1]=1.0
pps[ndv+1]=0.0

for k in ksrev
  krond = k==k_re ? 1.0 : 0.0
  js[k-1] = js[k] - krond
  pps[k-1] = js[k-1] * myτ / (myinput-Vks[k])
end

r0 = inv(dv*sum(pps))

#
_ = let plt = plot(;xlabel="membrane potential",ylabel="prob. density",
    leg=:topleft)
  histogram!(plt,myrec1.state_now[:];nbins=100,normalize=true,
    label="simulation",color=colorant"orange")
  plot!(plt,Vks,r0.*pps;label="Fokker-Plank",
   color=colorant"blue",linewidth=3)
end

# Let's do an f-I curve with this

function f_lif_num(h_in;dt=1E-4,Ttot=10.0)
  in_const = S.IFInputCurrentConstant(h_in)
  mypop = S.Population(ps,(conn_in,in_const))
  myntw = S.RecurrentNetwork(dt,mypop)
  rec = S.RecCountSpikes(ps,dt)
  # Run!
  times = (0:myntw.dt:Ttot)
````

initial conditions

````@example if_stationary_voltage_density
  ps.state_now[1] = v_r
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
    ps[k-1] = js[k-1] * myτ / (h_in-Vks[k])
  end
  return inv(dv*sum(ps))
end

#
_ = let Ttot = 50.0
  nsim = 40
  nfp = 200
  inputs_sim = range(vth+0.01,80.0;length=nsim)
  inputs_fp = range(vth+0.01,80.0;length=nfp)
  y1 = @showprogress map(in->f_lif_num(in;Ttot=Ttot),inputs_sim)
  y2 = map(in->f_lif_semianalitic(in),inputs_fp)
  plt = plot(leg=:bottomright,xlabel="input current",ylabel="stationary rate")
  scatter!(plt,inputs_sim,y1;
    label="simulation",color=colorant"orange")
  plot!(plt,inputs_fp,y2; label="Fokker-Planck",linewidth=2,
      color=colorant"blue" )
end
````

# Voltage density in the presence of input noise

The difference here is that the input is noisy, with a Gaussian distribution.

````@example if_stationary_voltage_density
const Ttot = 500.

const myinput = 8.5 #9.5
const myσ = 2.0 # 10.0
const in_normal = S.IFInputCurrentNormal(myinput,myσ)


const mypop = S.Population(ps,(conn_in,in_normal))
const myntw = S.RecurrentNetwork(dt,mypop)

const rec1_twup = 50.0 # further away from initial conditions
const myrec3 = S.RecStateNow(ps,10,dt,Ttot;Tstart=rec1_twup)
const myrec4 = S.RecCountSpikes(ps,dt)

times = (0:myntw.dt:Ttot)
nt = length(times)
````

initial conditions

````@example if_stationary_voltage_density
ps.state_now[1] = v_r

S.reset!.([myrec3,myrec4])

@showprogress for (k,t) in enumerate(times)
  S.dynamics_step!(t,myntw)
  myrec3(t,k,myntw)
  myrec4(t,k,myntw)
end
r0 = S.get_mean_rates(myrec4,dt,Ttot)[1]
@info r0
@info mean_and_std(myrec3.state_now[:])
#
_ = let tshow = (3.,20.)
  tshow = tshow .+ rec1_twup
  ts = myrec3.times
  y = myrec3.state_now[1,:]
  idx_keep = findall(t->tshow[1]<= t <= tshow[2],ts)
  plot(ts[idx_keep],y[idx_keep];leg=false,linewidth=1,xlabel="time (s)",
    ylabel="membrane potential (mV)",color=:black)
end
````

now, use Fokker-Planck theory and Richardson method
it is hardcoded for now... maybe later I will convert it to
a function

````@example if_stationary_voltage_density
const dv = 1E-3
const v_lb = v_r - 2.0
const ndv = round(Int64,(vth - v_lb)/dv)
const k_re = round(Int64,(v_r - v_lb)/dv)
const ks = collect(0:ndv)
const Vks = @. v_lb + ks * dv
const js = fill(NaN,ndv+1)
const pps = fill(NaN,ndv+1)

const ksrev = collect(ndv+1:-1:2)

js[ndv+1]=1.0
pps[ndv+1]=0.0

for k in ksrev
  krond = k == k_re ? 1.0 : 0.0
  js[k-1] = js[k] - krond
  Gk = (Vks[k]-myinput)/(myσ^2)
  Hk = myτ*js[k]/(myσ^2)
  Ak = exp(dv*Gk)
  Bk = Gk==0 ? dv*Hk : Hk/Gk*(Ak-1)
  pps[k-1] = pps[k]*Ak + Bk
end

r0 = inv(dv*sum(pps))

_ = let plt = plot(;xlabel="membrane potential",ylabel="prob. density",
    leg=:topleft)
  histogram!(plt,myrec3.state_now[:];
    nbins=100,normalize=true,label="simulation",
    color=colorant"orange")
  plot!(plt,Vks,r0.*pps;color=colorant"blue",
    linewidth=2,label="Fokker-Plank")
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

