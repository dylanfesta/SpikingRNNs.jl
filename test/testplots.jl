using LinearAlgebra: getindex
using Distributions: eltype, length
push!(LOAD_PATH, abspath(@__DIR__,".."))
using Pkg
pkg"activate ."

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) ; plotlyjs()
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using FFTW

function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end


##


myfunc(t::Real,β::Real) = t <= 0.0 ? 0.0 : β * exp(-β*t) 
myffou(ω::Real,β::Real) = β / (im*2π*ω + β)

function do_the_four(v::Vector{<:Real})
  N = length(v)
  exppart(n,k) = exp(-im*2π*k*n/N)
  ret = map( k -> mapreduce(n->v[n+1]*exppart(n,k), + , 0:N-1 ), 0:N-1 )
  return ret
end

mydt = 0.002
myT = 10.
myts = S.get_times(mydt,myT)
myfreqsh = S.get_frequencies_centerzero(mydt,myT)

myβ=2.0
test1 = fft(myfunc.(myts,myβ)) .* mydt

plot(t->myβ*exp(-t*myβ),myts)

test0 = let fx = myfunc.(myts,myβ)
  do_the_four(fx) .* mydt
end

test2 = myffou.(myfreqsh,myβ) |> fftshift

plot(real.(test0))
plot!(real.(test1))
plot!(real.(test2))


plot(imag.(test0))
plot!(imag.(test1))
plot!(imag.(test2);leg=false)



##
test_b1 = fft(myfunc.(myts,myβ)) .* mydt
test_b2 = fft(vcat(zeros(3000),myfunc.(myts,myβ)) ) .* mydt

plot( vcat(myfunc.(myts,myβ),myfunc.(myts,myβ)) )

plot(real.(test_b2))