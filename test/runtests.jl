using SpikingRNNs ; global const S=SpikingRNNs
using Test
using LinearAlgebra,SparseArrays,Statistics

@testset "SpikingRNNs.jl" begin
    # Write your tests here.

    m,n = (800,1000)
    ptest = 0.001
    μtest = 3.0
    σtest = 0.5
    wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest;exact=false)
    wtestvals = nonzeros(wtest)
    @test isapprox(mean(wtestvals),μtest;atol=0.1)
    @test isapprox(std(wtestvals),σtest;atol=0.1)
    wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest;exact=true)

end
