# First passage time density (and stationary rate, and ISI statistics)
# for a single LIF neuron without refractoriness!

push!(LOAD_PATH, abspath(@__DIR__,".."))
using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using FFTW
using ProgressMeter

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

# Time parameters
dt = 1E-4
Ttot = 5.
ntimes = floor(Int64,Ttot/dt)+1 
# LIF neuron parameters
myτ = 0.5
v_th = 10.
v_r = -5.0
τrefr = 0.0 # refractoriness
τpcd = 0.2 # post synaptic current decay
myinput = 10.1
myσ = 0.5
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
krec = 10
myrec1 = S.RecStateNow(myps,krec,dt,Ttot)
myrec2 = S.RecCountSpikes(myps,dt)

## Run

function one_run!(v_out,v_t0)
  # initial condition
  times = (0:dt:Ttot)
  myps.state_now[1] = v_t0
  S.reset!.([myrec1,myrec2])
  S.reset_spikes!(myps)
  for (k,t) in enumerate(times)
    myrec1(t,k,myntw)
    myrec2(t,k,myntw) # count spikes first
    if myrec2.spikecount[1] == 1
      v_out .= myrec1.state_now[1,:]
      return nothing
    end
    S.dynamics_step!(t,myntw)
  end
  @warn "Spike not reached! Consider iterating for longer time!"
end

# this function focuses on the time density only

function run_to_spike!(v_t0;Ttot=500.0)
  recS = S.RecCountSpikes(myps,dt)
  times = (0:dt:Ttot)
  myps.state_now[1] = v_t0
  S.reset!(recS)
  S.reset_spikes!(myps)
  for (k,t) in enumerate(times)
    recS(t,k,myntw)
    if recS.spikecount[1] == 1
      return t
    end
    S.dynamics_step!(t,myntw)
  end
  @warn "Spike not reached! Consider iterating for longer time!"
  return NaN
end


## compute and visualize the full densities 

Nsampl = 800
ntrec = size(myrec1.state_now,2)
v_all = Matrix{Float64}(undef,ntrec,Nsampl)

v_t0 = 1
@showprogress for v in eachslice(v_all;dims=2)
  one_run!(v,v_t0)
end

##
dv = 1E-1
v_lb = v_r - 3.0
ndv = round(Int64,(v_th - v_lb)/dv)
ks = collect(0:ndv) 
Vs = @. v_lb + ks * dv 
nVs=length(Vs)

fp_density_unnorm=Matrix{Float64}(undef,nVs-1,ntrec)

for (vt,dens) in zip(eachslice(v_all;dims=1),
    eachslice(fp_density_unnorm;dims=2))
  dens .= fit(Histogram,vt,Vs).weights
end

_ = let y = midpoints(Vs)
  ts_temp = myrec1.times
  dts = ts_temp[2]-ts_temp[1]
  ts = range(0;length=length(ts_temp),step=dts)
  z = log.(fp_density_unnorm)
  heatmap(ts,y,z ; xlabel="time",ylabel="membrane potential")
end


## Compute and visualize the spiketime density only

Nsampl = 1_000
v_t0 = 1
t_spike = @showprogress map(_->run_to_spike!(v_t0), 1:Nsampl)

_ = let dt = 0.1, Ttot=6.0,
  times = 0.0:dt:Ttot
  h=fit(Histogram,t_spike,times)
  h=normalize(h)
  plot(h; xlabel="first spiketime (s)",ylabel="density")
end


## now, use Fokker-Planck theory and Richardson method
# it is hardcoded for now... maybe later I will convert it to 
# a function
dv = 5E-3
v_lb = v_r - 2.0
ndv = floor(Int64,(v_th - v_lb)/dv)
k_v0 = floor(Int64,(v_t0 - v_lb)/dv)+1
ks = collect(0:ndv) 
Vs = @. v_lb + ks * dv 
ksrev = collect(ndv+1:-1:2)

# psifun(V) = steep_exp * exp((V-v_expt)/steep_exp)
psifun(V) = 0.0

function first_passage_stuff(ω)
  jf,jz,pf,pz =ntuple(_->fill(NaN+im*NaN,ndv+1),4)
  first_passage_stuff!(jf,jz,pf,pz,ω)
end

function first_passage_stuff!(jf,jz,pf,pz,ω)
  jf[ndv+1]=1.0
  for v in (jz,pf,pz)
    v[ndv+1]=0.0
  end
  for k in ksrev
    kron_v0 = k==k_v0 ? 1.0 : 0.0
    jf[k-1] = jf[k] + 2π*dv*im*ω*pf[k]
    jz[k-1] = jz[k] + 2π*dv*im*ω*pz[k] - kron_v0 # exp(im*ω*t0) with t0=0
    Gf = (Vs[k]-myinput-psifun(Vs[k]))/(myσ^2); Gz = Gf
    Ak = exp(dv*Gf)
    Hf = myτ*jf[k]/(myσ^2)
    Hz = myτ*jz[k]/(myσ^2)
    pf[k-1] = Ak==1.0 ? pf[k] + Hf*dv : pf[k]*Ak + Hf/Gf*(Ak-1) 
    pz[k-1] = Ak==1.0 ? pz[k] + Hz*dv : pz[k]*Ak + Hz/Gz*(Ak-1) 
  end
  return -jz[1]/jf[1]
end


#firstpass=first_passage_stuff(6.0)

## from Fourier to time domain 

# frequencies for the Fourier transform
# from 0 to 1/dt - 1/T in steps of 1/T
function get_frequencies(dt::Real,T::Real)
  dω = inv(T)
  ωmax = inv(dt)
  f = 0.0:dω:ωmax
  return f
end

@inline function get_times(dt::Real,T::Real)
  return (0.0:dt:(T-dt))
end

function get_first_passage_density(dt,Tmax;c1::Integer=1,c2::Real=1.0)
  newTmax = c1*Tmax
  newdt = c2*dt
  mytaus = get_times(newdt,newTmax)
  nkeep = div(length(mytaus),c1)
  ωs = get_frequencies(newdt,newTmax)
  jf,jz,pf,pz =ntuple(_->fill(NaN+im*NaN,ndv+1),4)
  fω = map(ω->first_passage_stuff!(jf,jz,pf,pz,ω),ωs)
  fω[2:end] .*= 2   # WHY ??? 
  ft =real.(ifft(fω)) ./ newdt
  return mytaus[1:nkeep],ft[1:nkeep]
end

##

first_passage_out=get_first_passage_density(1E-1,5.0;c1=2,c2=0.5)


_ = let dt = 0.1, Ttot=8.0,
  times = 0.0:dt:Ttot,
  fpo=first_passage_out,
  h=fit(Histogram,t_spike,times)
  h=normalize(h)
  plt=plot(;xlabel="first spiketime (s)",ylabel="density")
  plot!(plt,h)
  plot!(fpo[1],fpo[2];color=:white,linewidth=2)
end

##

function get_full_density(dt,Tmax;c1::Integer=1,c2::Real=1.0)
  newTmax = c1*Tmax
  newdt = c2*dt
  mytaus = get_times(newdt,newTmax)
  nkeep = div(length(mytaus),c1)
  ωs = get_frequencies(newdt,newTmax)
  nωs = length(ωs)
  jf,jz,pf,pz =ntuple(_->fill(NaN+im*NaN,ndv+1),4)
  pfou = fill(NaN+im*NaN,ndv+1,nωs)
  pout = similar(pfou,Float64)
  for (ω,pfouω) in zip(ωs,eachslice(pfou;dims=2))
    fω=first_passage_stuff!(jf,jz,pf,pz,ω)
    if ω != 0
      fω *= 2   # WHY ???
    end
    @. pfouω = fω * pf + pz
  end
  for (poutt,pfouω) in zip(eachslice(pout;dims=1),eachslice(pfou;dims=1))
     poutt .= real.(ifft(pfouω)) ./ newdt
  end
  return mytaus[1:nkeep],pout[:,1:nkeep]
end

# when dt is too small, some weird numerical artifacts pop up :-/
_ = let dt=0.1 , Tmax = 6, 
  dat = get_full_density(dt,Tmax;c1=2,c2=1.0) 
  p = dat[2]
  ts = dat[1]
  y = Vs 
  z = abs.(p)
  @show size(ts) size(y) size(z)
  heatmap(ts,y,z ; xlabel="time",ylabel="membrane potential")
end
