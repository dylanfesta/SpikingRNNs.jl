using SpikingRNNs ; global const S=SpikingRNNs
using Test
using LinearAlgebra,SparseArrays,Statistics
using Random ; Random.seed!(0)


# utility test functions

# connections
function onesparsemat(w::Real)
  mat=Matrix{Float64}(undef,1,1) ; mat[1,1]=w
  return sparse(mat)
end

##

@testset "Connectivity matrices" begin
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

@testset "2D rate model" begin
    
  pope = S.PopRateReLU(1,1.,1.)
  popi = S.PopRateReLU(1,1.,1.)
  pse = S.PSRate(pope)
  psi = S.PSRate(popi)
  S.reset_input!.((pse,psi))

  (w_ee,w_ie,w_ei,w_ii) = let w = 20. , k = 1.1
    onesparsemat.((w,w,-k*w,-k*w))
  end

  conn_ee = S.ConnectionRate(pse,w_ee,pse)
  conn_ei = S.ConnectionRate(pse,w_ei,psi)
  conn_ie = S.ConnectionRate(psi,w_ie,pse)
  conn_ii = S.ConnectionRate(psi,w_ii,psi)

  # inputs
  in_e = S.PopInputStatic(pse,[0.,])
  in_i = S.PopInputStatic(psi,[0.,])

  # initial conditions
  pse.state_now[1] = S.ioinv(10.0,pse)
  psi.state_now[1] = S.ioinv(5.0,pse)

  dt = 1E-4
  T = 5.0
  times = 0:dt:T 
  ntimes = length(times)
  mynetwork = S.RecurrentNetwork(dt,(pse,psi),(in_e,in_i),
    (conn_ee,conn_ie,conn_ei,conn_ii) )

  e_out = Vector{Float64}(undef,ntimes)
  i_out = Vector{Float64}(undef,ntimes)

  for (k,t) in enumerate(times) 
    e_out[k] = S.iofunction(pse.state_now[1],pse)
    i_out[k] = S.iofunction(psi.state_now[1],psi)
    # rate model with constant input  does not really depend on absolute time (first argument)
    S.dynamics_step!(0.0,mynetwork)  
  end

  @test all(e_out .>= 0.0)
  @test all(isfinite.(e_out))
  @test all(i_out .>= 0.0)
  @test all(isfinite.(i_out))
  # I expect aplification, let's say that 1/3 of elements are above the starting rate
  # but the last is below
  @test i_out[end]<i_out[1]
  @test count(i_out .> i_out[1])/length(i_out) > 0.333
  @test count(e_out .> e_out[1])/length(e_out) > 0.333

end

@testset "single LIF neuron" begin
  dt = 5E-4
  myτ = 0.1
  vth = 12.
  v_r = -6.123
  τrefr = 0.5
  τpcd = 1E10
  e1 = S.PopLIF(1,myτ,vth,v_r,τrefr,τpcd)
  pse1 = S.PSLIF(e1)

  # one static input 
  my_input = 14.0
  pse_in = S.PopInputStatic(pse1,[my_input,])

  # empty connection (to avoid errors)
  mywmat = sparse(zeros(Float64,(1,1)))
  conn_ee = S.ConnectionLIF(pse1,mywmat,pse1)
  # that's it, let's make the network
  myntw = S.RecurrentNetwork(dt,(pse1,),(pse_in,),(conn_ee,) )

  Ttot = 10.0 
  times = (0:myntw.dt:Ttot)
  nt = length(times)
  pse1.state_now[1] = v_r
  myvs = Vector{Float64}(undef,nt)
  myfiring = BitVector(undef,nt)
  for (k,t) in enumerate(times)
    S.dynamics_step!(t,myntw)
    myvs[k] = pse1.state_now[1]
    myfiring[k]=pse1.isfiring[1]
  end

  # period of first spike
  @test isapprox(S.expected_period_norefr(e1,my_input), times[findfirst(myfiring)] ;
    atol = 0.02)

  # number of spikes
  myper_postrest = S.expected_period_norefr(e1.τ,0.0,e1.v_threshold,my_input)
  nspk_an = floor(Ttot/(e1.τ_refractory + myper_postrest ) )
  @test isapprox(nspk_an,count(myfiring) ; atol=2)
end