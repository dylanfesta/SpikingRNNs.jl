# First passage time density (and stationary rate, and ISI statistics)
# for a single LIF neuron without refractoriness!

push!(LOAD_PATH, abspath(@__DIR__,"..","src"))
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
dt = 1E-3
Ttot = 6.
ntimes = floor(Int64,Ttot/dt)+1 
# LIF neuron parameters
myτ = 0.5
v_th = 10.
v_r = -5.0
τrefr = 0.0 # refractoriness
τpcd = 0.2 # post synaptic current decay
myinput = 9.5
myσ = 1.0
mynt = S.NTLIF(myτ,v_th,v_r,τrefr,τpcd)
myps = S.PSLIF(mynt,1)

## inputs here
# one periodic input (freq is the inverse of period)
ωin = inv(2.0)
infun = function(t)
  α = 1.
  return myinput + α*sin(ωin*t*2π)
end
in_state = S.PSSimpleInput(S.InputSomeFunction(infun))
conn_in = S.BaseConnection(onesparsemat(1.0))
# ... plus noise
in_state_noise = S.PSSimpleInput(S.InputIndependentNormal(myσ))
##
mypop = S.Population(myps,(conn_in,S.FakeConnection()),
    (in_state,in_state_noise))


## that's it, let's make the network

myntw = S.RecurrentNetwork(dt,(mypop,))

## Recorder
krec = 10
rec_state = S.RecStateNow(myps,krec,dt,Ttot)
rec_spikes = S.RecCountSpikes(myps,dt)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# initial conditions
myps.state_now[1] = v_r

S.reset!.([rec_state,rec_spikes])

for (k,t) in enumerate(times)
  rec_state(t,k,myntw)
  rec_spikes(t,k,myntw)
  S.dynamics_step!(t,myntw)
end
r0 = S.get_mean_rates(rec_spikes,dt,Ttot)[1]
##
_ = let plt=plot(;xlabel="time",ylabel="voltage",leg=false)
  plot!(plt,rec_state.times,rec_state.state_now[1,:],linewidth=2)
end
##

function simulation_linear_response_coef(ω,E;dt=1E-3,Ttot::Float64=1000.0)
  # network without extra input
  in_noise = S.PSSimpleInput(S.InputIndependentNormal(myσ))
  in_constant = S.PSSimpleInput(S.InputSimpleOffset(myinput))
  mypop = S.Population(myps,
      (S.FakeConnection(),S.FakeConnection()),
      (in_constant,in_noise))

  rec_spikes = S.RecCountSpikes(myps,dt)
  times = (0:dt:Ttot)
  # initial conditions
  myps.state_now[1] = v_r
  ntw = S.RecurrentNetwork(dt,(mypop,))
  S.reset!(rec_spikes)
  @showprogress "1/2 --> " for (k,t) in enumerate(times)
    rec_spikes(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
  r0 = S.get_mean_rates(rec_spikes,dt,Ttot)[1]
  # network with extra input
  infun = function(t)
    return myinput + E*sin(ω*t*2π)
  end
  in_func = S.PSSimpleInput(S.InputSomeFunction(infun))
  conn_in = S.BaseConnection(onesparsemat(1.0))
  mypop = S.Population(myps,
      (conn_in,S.FakeConnection()),
      (in_func,in_noise))

  myps.state_now[1] = v_r
  ntw = S.RecurrentNetwork(dt,(mypop,))
  S.reset!(rec_spikes)
  @showprogress "2/2 --> " for (k,t) in enumerate(times)
    rec_spikes(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
  r1 = S.get_mean_rates(rec_spikes,dt,Ttot)[1]
  return (r1-r0)/E
end

##

t_period = range(0.1,2;length=20)
lrs = map(t_period) do _t
  simulation_linear_response_coef(inv(_t),10.0;Ttot=1500.0)
end
lrs2 = map(t_period) do _t
  simulation_linear_response_coef(inv(_t),30.0;Ttot=1500.0)
end

plot(t_period,[lrs lrs2];label=["E=10" "E=30"], marker=:star)

############################
### now follow the approach of Richardson et al
##
# start with stationary case

dv = 1E-3
v_lb = v_r - 3.0
ndv = round(Int64,(v_th - v_lb)/dv)
k_re = round(Int64,(v_r - v_lb)/dv)
ks = collect(0:ndv) 
Vks = @. v_lb + ks * dv

j0s = fill(NaN,ndv+1)
p0s = fill(NaN,ndv+1)

ksrev = collect(ndv+1:-1:2)

j0s[ndv+1]=1.0
p0s[ndv+1]=0.0

for k in ksrev
  krond = k==k_re ? 1.0 : 0.0
  j0s[k-1] = j0s[k] - krond
  Gk = (Vks[k]-myinput)/(myσ^2)
  Hk = myτ*j0s[k]/(myσ^2)
  Ak = exp(dv*Gk)
  Bk = Gk==0 ? dv*Hk : Hk/Gk*(Ak-1)
  p0s[k-1] = p0s[k]*Ak + Bk
end

r0 = inv(dv*sum(p0s)) 
P0s = r0 .* p0s

## now the non-stationary linear addendum

function theory_linear_response!(jr,je,pr,pe,ω::Real;E=1.0)
  jr[ndv+1] = 1.0
  je[ndv+1] = 0.
  pr[ndv+1] = 0.
  pe[ndv+1] = 0.
  for k in ksrev
    krond = k==k_re ? 1.0 : 0.0
    jr[k-1] = jr[k] + im*2π*ω*pr[k] - krond
    je[k-1] = je[k] + im*2π*ω*pe[k]
    G = (Vks[k]-myinput)/(myσ^2)
    Hr = myτ*jr[k]/(myσ^2)
    He = (myτ*je[k]-P0s[k])/(myσ^2)
    Ak = exp(dv*G)
    pr[k-1] = G==0 ? pr[k] + Hr*dv : pr[k]*Ak + Hr/G*(Ak-1) 
    pe[k-1] = G==0 ? pe[k] + He*dv : pe[k]*Ak + He/G*(Ak-1) 
    #@show pe[k] 
    #@show pe[k-1]
    @assert isfinite(pe[k-1])
  end
  #return je
  AEω = (-je[1]/jr[1])
  Pfull = @. r0*p0s + E*AEω + E*pe
  return AEω ,Pfull
end

_jr,_je,_pr,_pe =ntuple(_->similar(p0s,ComplexF64),4) 

_Aω,_Pfullω = theory_linear_response!(_jr,_je,_pr,_pe,inv(6.0))


_ = let y = real.(_Pfullω),
  x=Vks
  plot(x,y)
end

