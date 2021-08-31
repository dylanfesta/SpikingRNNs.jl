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
Ttot = 30.
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
hw_in = onesparsemat(myinput)
conn_in = S.BaseConnection(hw_in)
in_type = S.InputSimpleOffset()
in_state = S.PSSimpleInput(in_type)
mypop = S.Population(myps,(conn_in,),(in_state,))


## that's it, let's make the network
myntw = S.RecurrentNetwork(dt,(mypop,))

## Recorder
myrec1 = S.RecStateNow(myps,10,dt,Ttot)
myrec2 = S.RecCountSpikes(myps)

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
r0 = S.get_mean_rates(myrec2,Ttot)[1]
##
plot(myrec1.times,myrec1.state_now[1,:])
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
  ps[k-1] = js[k-1] * mynt.τ / (myinput-Vks[k]+0)
end

r0 = inv(dv*sum(ps))

##
histogram(myrec1.state_now[:];nbins=100,normalize=true)
plot!(Vks,r0.*ps;color=:white,linewidth=2)
## can I use this for the f-I curve?

function f_lif_num(h_in;dt=1E-4,Ttot=10.0)
  in_state = S.PSSimpleInput(S.InputSimpleOffset())
  conn_in = S.BaseConnection(onesparsemat(h_in))
  mypop = S.Population(myps,(conn_in,),(in_state,))
  myntw = S.RecurrentNetwork(dt,(mypop,))
  rec = S.RecCountSpikes(myps)
  ## Run!
  times = (0:myntw.dt:Ttot)
  # initial conditions
  myps.state_now[1] = v_r
  S.reset!(rec)
  for (k,t) in enumerate(times)
    S.dynamics_step!(t,myntw)
    rec(t,k,myntw)
  end
  r0 = S.get_mean_rates(rec,Ttot)[1]
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
    ps[k-1] = js[k-1] * mynt.τ / (h_in-Vks[k]+0)
  end
  return inv(dv*sum(ps))
end

##
_ = let Ttot = 50.0
  nin = 80
  inputs = range(0.0,80.0;length=nin)
  y1 = similar(inputs)
  y2 = similar(inputs)
  @showprogress for i in eachindex(inputs)
    y1[i]=f_lif_num(inputs[i];Ttot=Ttot)
    y2[i]=f_lif_semianalitic(inputs[i])
  end
  plot(inputs,[y1 y2]; 
     label=["simulation" "Fokker-Planck"],
     linewidth=2,marker=:star,leg=:bottomright)
end

