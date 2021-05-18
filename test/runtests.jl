using SpikingRNNs ; global const S=SpikingRNNs
using Test
using LinearAlgebra,SparseArrays,Statistics
using Random ; Random.seed!(0)

@testset "SpikingRNNs.jl" begin
    # Write your tests here.

    m,n = (800,1000)
    μtest = 3.0
    ptest = 0.1
    σtest = 0.5
    wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest;exact=false)
    wtestvals = nonzeros(wtest)
    @test isapprox(mean(wtestvals),μtest;atol=0.1)
    @test isapprox(std(wtestvals),σtest;atol=0.1)
    wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest;exact=true)
    @test all(isapprox.(sum(wtest;dims=2),μtest;atol=0.15))
    m,n = (80,100)
    μtest = -3.0
    ptest = 0.333
    wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest;exact=false)
    wtestvals = nonzeros(wtest)
    @test isapprox(mean(wtestvals),μtest;atol=0.1)
end
