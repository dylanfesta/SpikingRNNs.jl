#=
# Single LIF neuron, not refractory, first passage time density

I simulate a single neuron that receives a periodic input, plus noise
and I then compare analytic and numeric results for its distribution of 
membrane potentials. This is the first passage time density.

See papers by Richardson et al
=#

## #src
push!(LOAD_PATH, abspath(@__DIR__,"..","src"))
using LinearAlgebra,Statistics,StatsBase
using Plots,NamedColors ; theme(:default)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using FFTW
using ProgressMeter

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

# ## Time parameters
const dt = 1E-3
const Ttot = 20.
const ntimes = floor(Int64,Ttot/dt)+1 
# ## IF neuron parameters
const τ = 0.5
const cap = τ
#const cap = 1.0
const v_th = 10.
const v_r = -5.0
const vleak = 0.0
const τrefr = 0.0 # refractoriness
const ps = S.PSIFNeuron(1,τ,cap,v_th,v_r,vleak,τrefr);

## #src
#= ## Inputs

The first input is Gaussian noise. We add the constant external input
as the mean of the Gaussian.
=#
const myσ = 10.0
const myinput = 9.5
const in_normal = S.IFInputCurrentNormal(myinput,myσ);
const conn_in = S.ConnectionIFInput([1.]);

#=
The second input is a sinusoidal function with mean zero.

Note that we could also have zero mean Gaussian noise, and a positive offset here.
=#
const ωin = inv(2.0)
infun = function(t)
  return sin(ωin*t*2π)
end
const in_sin = S.IFInputCurrentFunScalar(infun);

# Build network

const pop = S.Population(ps,(conn_in,in_normal),(conn_in,in_sin))

const ntw = S.RecurrentNetwork(dt,pop);

# ## Define recorders: spikes and internal state

const krec = 10
const rec_state = S.RecStateNow(ps,krec,dt,Ttot)
const rec_spikes = S.RecCountSpikes(ps,dt)

# ## Run the network
const times = (0:ntw.dt:Ttot)
const nt = length(times)
# initial conditions
ps.state_now[1] = v_r

S.reset!.([rec_state,rec_spikes])
for (k,t) in enumerate(times)
  rec_state(t,k,ntw)
  rec_spikes(t,k,ntw)
  S.dynamics_step!(t,ntw)
end
r0 = S.get_mean_rates(rec_spikes,dt,Ttot)[1]
## #src

_ = let plt=plot(;xlabel="time",ylabel="voltage",leg=false)
  plot!(plt,rec_state.times,rec_state.state_now[1,:],linewidth=2)
end
## #src
#= ## Simulate at different input frequencies

In the function below I define a network and run it, either with sinusoidal signal or 
without. I compare the rate difference in the two cases.
=#
function simulation_linear_response_coef(ω,E;dt=1E-3,Ttot::Float64=1000.0)
  # without sinusoidal #src
  in_normal = S.IFInputCurrentNormal(myinput,myσ);
  conn_in = S.ConnectionIFInput([1.]);
  mypop = S.Population(ps,(conn_in,in_normal))
  rec_spikes = S.RecCountSpikes(ps,dt)
  S.reset!(rec_spikes)
  S.reset!(ps)
  times = (0:dt:Ttot)
  ps.state_now[1] = v_r
  ntw = S.RecurrentNetwork(dt,mypop)
  for (k,t) in enumerate(times)
    rec_spikes(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
  r0 = S.get_mean_rates(rec_spikes,dt,Ttot)[1]
  # with sinusoidal #src
  infun = function(t)
    return myinput + E*sin(ω*t*2π)
  end
  in_sin = S.IFInputCurrentFunScalar(infun)
  mypop = S.Population(ps,(conn_in,in_normal),(conn_in,in_sin))
  S.reset!(rec_spikes)
  S.reset!(ps)
  ps.state_now[1] = v_r
  ntw = S.RecurrentNetwork(dt,(mypop,))
  S.reset!(rec_spikes)
  for (k,t) in enumerate(times)
    rec_spikes(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
  r1 = S.get_mean_rates(rec_spikes,dt,Ttot)[1]
  return (r1-r0)/E
end

# ## Test it!

t_period = range(0.1,2;length=20)
lrs = @showprogress "1/2  " map(t_period) do _t
  simulation_linear_response_coef(inv(_t),10.0;Ttot=1500.0)
end
lrs2 = @showprogress "2/2  " map(t_period) do _t
  simulation_linear_response_coef(inv(_t),30.0;Ttot=1500.0)
end

plot(t_period,[lrs lrs2];label=["E=10" "E=30"], marker=:star)

## #src

#=
##  Now follow the approach of Richardson et al

start with stationary case

Vector of membrane potentials
=#

const dv = 1E-3
const v_lb = v_r - 3.0
const ndv = round(Int64,(v_th - v_lb)/dv)
const k_re = round(Int64,(v_r - v_lb)/dv)
const ks = collect(0:ndv) 
const Vks = collect(range(v_lb,v_th;length=ndv+1))

const j0s = fill(NaN,ndv+1)
const p0s = fill(NaN,ndv+1)

const ksrev = collect(ndv+1:-1:2)

j0s[ndv+1]=1.0
p0s[ndv+1]=0.0


for k in ksrev
  krond = k==k_re ? 1.0 : 0.0
  j0s[k-1] = j0s[k] - krond
  Gk = (Vks[k]-myinput)/(myσ^2)
  Hk = τ*j0s[k]/(myσ^2)
  Ak = exp(dv*Gk)
  Bk = Gk==0 ? dv*Hk : Hk/Gk*(Ak-1)
  p0s[k-1] = p0s[k]*Ak + Bk
end

r0 = inv(dv*sum(p0s)) 
P0s = r0 .* p0s;

# ### Non-stationary linear addendum

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
    Hr = τ*jr[k]/(myσ^2)
    He = (τ*je[k]-P0s[k])/(myσ^2)
    Ak = exp(dv*G)
    pr[k-1] = G==0 ? pr[k] + Hr*dv : pr[k]*Ak + Hr/G*(Ak-1) 
    pe[k-1] = G==0 ? pe[k] + He*dv : pe[k]*Ak + He/G*(Ak-1) 
    @assert isfinite(pe[k-1])
  end
  #return je #src
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


## Publish it... NOT! #src
#using Literate #src
#Literate.markdown(@__FILE__,"docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/SpikingRNNs.jl/blob/master") #src