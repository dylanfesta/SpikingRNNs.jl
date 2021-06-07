using LinearAlgebra: getindex
using Distributions: eltype
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

n = 6
myβ = 1.1
mywdistr=Uniform(0.2,0.4)
mywmat = sprand(n,n,0.5,n->rand(mywdistr,n))
myin = rand(Uniform(0.15,0.2),n)
tfake = NaN
p1 = S.PopulationHawkesExp(n,myβ)
ps1 = S.PSHawkes(p1)
conn1 = S.ConnectionHawkes(ps1,mywmat,ps1)
# rates to test
p1_in = S.PopInputStatic(ps1,myin)
myntw = S.RecurrentNetwork(tfake,(ps1,),(p1_in,),(conn1,) )

#
nspikes = 500_000
# initialize
S.hawkes_initialize!(myntw)

my_act = Vector{Tuple{Int64,Float64}}(undef,nspikes)
for k in 1:nspikes
  S.dynamics_step!(tfake,myntw)
  idx_fire = findfirst(ps1.isfiring)
  t_now = ps1.time_now[1]
  my_act[k] = (idx_fire,t_now)
end

myspikes_all = [S.hawkes_get_spiketimes(i,my_act) for i in 1:n]
myTmax = maximum(x->x[end],myspikes_all)+eps() 
@info "Total duration $(round(myTmax;digits=1)) s"
rates_all =  map(x->length(x)/myTmax, myspikes_all )
@info "Rates are $(round.(rates_all;digits=2)) "

##
# now mean rate, covariance, etc

# covariance numerical

mydt = 0.2
myτmax = 80.0
mytaus = S.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = S.covariance_density_numerical(myspikes_all,mydt,myτmax;verbose=true)

##

# numerical cov densities
plot(mytaus[2:end-1],cov_num[29][2][2:end-1] ; linewidth = 2)
plot!(mytaus[2:end-1],cov_num[2][2][2:end-1] ; linewidth = 2)
plot!(mytaus[2:end-1],cov_num[12][2][2:end-1] ; linewidth = 2)

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
 Measured rates:  $(round.(rates_all;digits=2))
 Rates with Fourier :  $(round.(ratefou;digits=2))
"""

# now the main one...

D = diagm(0=>ratefou)

Cfou = Array{eltype(gfou_all[1,1])}(undef,n,n,length(myfreq))

for k in 1:length(myfreq) 
  Gfm =  getindex.(gfou_all_shift,k)
  # the ' operator corresponds to transpose and -ω 
  Cfou[:,:,k] = ( inv((I-Gfm)) * D * inv((I-Gfm')) ) .* inv(mydt)
end

# C analytic
C_ana = mapslices(v-> real.(ifft(v)) ,Cfou;dims=3)
_ = let _ = nothing 
  @info "Zero lag covariances"
  for cnum in cov_num
    (i,j) = cnum[1]
    println("numerical & Fourier :  $(round(cnum[2][1];digits=2)) & $(round(C_ana[i,j,1];digits=2))")
  end
  println()
end

##

_ = let idx = 4,
  cn = cov_num[idx],
  (i,j) = cn[1]
  @show (i,j)
  Cana_fix = copy(C_ana[i,j,2:ntaus])
  if i!=j
    Cana_fix .-= mydt^2 * sqrt(ratefou[i]*ratefou[j]) 
  end
  myplot = plot(mytaus[2:end],cn[2][2:end] ; linewidth = 2)
  plot!(myplot,mytaus[2:end],Cana_fix; linewidth = 2)
end




