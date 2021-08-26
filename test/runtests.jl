using SpikingRNNs ; global const S=SpikingRNNs
using Test
using LinearAlgebra,SparseArrays,Statistics,Distributions
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
  wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest)
  wtestvals = nonzeros(wtest)
  @test isapprox(mean(wtestvals),μtest;atol=0.1)
  @test isapprox(std(wtestvals),σtest;atol=0.1)
  @test tr(Diagonal(abs.(wtest))) == 0.0 # no autpses
  m,n = (80,100)
  μtest = -3.0
  ptest = 0.333
  wtest = S.sparse_wmat_lognorm(m,n,ptest,μtest,σtest;rowsum=123.0)
  @test all(isapprox.(sum(wtest;dims=2),-123.0;atol=0.001))

  wtest = S.sparse_wmat(m,n,ptest,-0.123)
  wvals = nonzeros(wtest)
  @test all( isapprox.(wvals,-0.123) )
  
end

@testset "2D rate model" begin

  neuron_e = S.NTReLU(1.,1.)
  neuron_i = S.NTReLU(1.,1.)
  pse  = S.PSRate(neuron_e,1)
  psi  = S.PSRate(neuron_i,1)

  (w_ee,w_ie,w_ei,w_ii) = let w = 20. , k = 1.1
    onesparsemat.((w,w,-k*w,-k*w))
  end

  conn_ee = S.BaseConnection(w_ee)
  conn_ei = S.BaseConnection(w_ei)
  conn_ie = S.BaseConnection(w_ie)
  conn_ii = S.BaseConnection(w_ii)

  pop_e = S.Population(pse,(conn_ee,conn_ei),(pse,psi))
  pop_i = S.Population(psi,(conn_ie,conn_ii),(pse,psi))

  # initial conditions
  pse.state_now[1] = S.ioinv(10.0,pse)
  psi.state_now[1] = S.ioinv(5.0,pse)

  dt = 1E-4
  T = 5.0
  times = 0:dt:T 
  ntimes = length(times)
  mynetwork = S.RecurrentNetwork(dt,(pop_e,pop_i))

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
  @test e_out[end]<e_out[1]
  @test count(i_out .> i_out[1])/length(i_out) > 0.333
  @test count(e_out .> e_out[1])/length(e_out) > 0.333

end

@testset "2D rate model with input" begin
  
  neuron_ei = S.NTReLU(1.,1.)
  psei  = S.PSRate(neuron_ei,2)

  wmat = sparse([ 2.  -3.
                  2.5  -0.5 ]) 
  input_mat = let ret=Matrix{Float64}(undef,2,1)
    ret.=[50.33,2.8]
    sparse(ret)
  end

  conn_rec = S.BaseConnection(wmat)                  

  fpoint = - inv(Matrix(wmat)-I)*input_mat
  ## input connection!
  in_type = S.InputSimpleOffset()
  in_state = S.PSSimpleInput(in_type)
  conn_in = S.BaseConnection(input_mat)
  pop_ei = S.Population(psei,(conn_rec,conn_in),(psei,in_state))
  ##
  dt = 1E-2
  T = 60.0
  times = 0:dt:T 
  ntimes = length(times)
  mynetwork = S.RecurrentNetwork(dt,(pop_ei,))

  ei_out = Matrix{Float64}(undef,2,ntimes)
  # initial conditions
  psei.state_now .= S.ioinv(10.0,psei)

  for (k,t) in enumerate(times) 
    ei_out[:,k] = S.iofunction.(psei.state_now,neuron_ei)
    # rate model with constant input  does not really depend on absolute time (first argument)
    S.dynamics_step!(mynetwork)  
  end
  @test all(isapprox.(ei_out[:,end],fpoint;atol=1E-1))

end



#=
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

@testset "Hawkes isolated" begin
  
  # isolated, non-interacting , processes 
  # with given input

  myβ = 0.5
  dt = 10E10 # useless
  myn = 40 
  p1 = S.PopulationHawkesExp(myn,myβ)
  ps1 = S.PSHawkes(p1)
  conn1 = S.ConnectionHawkes(ps1,sparse(zeros(myn,myn)),ps1)
  # rates to test
  myrates = rand(Uniform(0.5,4.0),myn)
  p1_in = S.PopInputStatic(ps1,myrates)
  myntw = S.RecurrentNetwork(dt,(ps1,),(p1_in,),(conn1,) )
  ##
  nspikes = 100_000
  # initialize
  tfake = NaN
  ps1.state_now .= 1E-2
  S.send_signal!(tfake,p1_in)
  # save activity as #idx_fired  , t_fired
  my_act = Vector{Tuple{Int64,Float64}}(undef,nspikes)
  for k in 1:nspikes
    S.dynamics_step!(tfake,myntw)
    idx_fire = findfirst(ps1.isfiring)
    t_now = ps1.time_now[1]
    my_act[k] = (idx_fire,t_now)
  end

  function hawkes_mean_rates(nneus,actv::Vector)
    t_end = actv[end][2]
    _f = function(i)
      return count(ac->ac[1]==i,actv)/t_end
    end
    return map(_f,1:nneus)
  end
  myrates_sim = hawkes_mean_rates(myn,my_act)
  @test all( isapprox.(myrates,myrates_sim ;atol=0.25))
end

=#