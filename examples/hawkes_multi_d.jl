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
mywdistr=Uniform(0.2,0.3)
mywmat = sprand(n,n,0.5,n->rand(mywdistr,n))
mywmat_nonsp = convert(Matrix{Float64},mywmat)
myin = rand(Uniform(0.1,0.5),n)
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

# rates with Fourier 
# analytic rate from fourier Eq between 6 and  7 in Hawkes
ratefou = let G0 =  mywmat_nonsp .* S.fou_interaction_kernel.(0,p1)
  inv(I-G0)*myin |> real
end 

@info "With Fourier, rates are $(round.(ratefou;digits=2)) Hz"

##
# covariance numerical

mydt = 0.2
myτmax = 80.0
mytaus = S.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = S.covariance_density_numerical(myspikes_all,mydt,myτmax)

##

# numerical cov densities
plot(mytaus[2:end],cov_num[1,1,2:end] ; linewidth = 2)
plot!(mytaus[2:end],cov_num[2,2,2:end] ; linewidth = 2)
plot!(mytaus[2:end],cov_num[2,5,2:end] ; linewidth = 2)

## now with Fourier!

# Pick eq 12 from Hawkes
function four_high_res(dt::Real,Tmax::Real) 
  k1 = 2
  k2 = 0.2
  n = size(mywmat_nonsp,1)
  myτmax,mydt = Tmax * k1, dt*k2
  mytaus = S.get_times(mydt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = S.get_frequencies_centerzero(mydt,myτmax)
  G_omega = map(mywmat_nonsp) do w
    ifftshift( w .* S.fou_interaction_kernel.(myfreq,p1))
  end
  D = Diagonal(ratefou)
  M = Array{ComplexF64}(undef,n,n,length(myfreq))
  Mt = similar(M,Float64)
  for i in eachindex(myfreq)
    G = getindex.(G_omega,i)
    # M[:,:,i] = (I-G)\(G*D+D*G'-G*D*G')/(I-G') 
    M[:,:,i] = (I-G)\D*(G+G'-G*G')/(I-G') 
  end
  for i in 1:n,j in 1:n
    Mt[i,j,:] = real.(ifft(M[i,j,:])) ./ mydt
  end
  return mytaus[1:nkeep],Mt[:,:,1:nkeep]
end


taush,Cfou=four_high_res(mydt,myτmax)



# numerical cov densities
plot(mytaus[2:end],cov_num[1,1,2:end] ; linewidth = 2)
plot!(taush[1:end],Cfou[1,1,1:end] ; linewidth=3, linestyle=:dash)


plot(mytaus[2:end],cov_num[2,2,2:end] ; linewidth = 2)
plot!(taush[1:end],Cfou[2,2,1:end] ; linewidth=3, linestyle=:dash)


plot(mytaus[2:end],cov_num[2,5,2:end] ; linewidth = 2)
plot!(taush[3:end],Cfou[2,5,3:end] ; linewidth=3, linestyle=:dash)


plot(mytaus[2:end],cov_num[1,6,2:end] ; linewidth = 2)
plot!(taush[3:end],Cfou[1,6,3:end] ; linewidth=3, linestyle=:dash)