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

  pop_e = S.Population(pse,(conn_ee,pse),(conn_ei,psi))
  pop_i = S.Population(psi,(conn_ie,pse),(conn_ii,psi))

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
 
  neuron_e_and_i = S.NTReLU(1.,1.)
  pse  = S.PSRate(neuron_e_and_i,1)
  psi  = S.PSRate(neuron_e_and_i,1)

  (w_ee,w_ie,w_ei,w_ii) = (2.,2.5,-3.,-0.5)
  wmat = [w_ee  w_ei 
          w_ie w_ii]
  make_connection(w)=S.BaseConnection(onesparsemat(w)) 
  conn_ee = make_connection(w_ee)
  conn_ei = make_connection(w_ei)
  conn_ie = make_connection(w_ie)
  conn_ii = make_connection(w_ii)

  # inputs
  h_e = 50.33
  h_i = 2.8
  h_vec = [h_e,h_i]

  fpoint = - inv(Matrix(wmat)-I)*h_vec

  ## input connection!
  in_state_e = S.PSSimpleInput(S.InputSimpleOffset(h_e))
  in_state_i = S.PSSimpleInput(S.InputSimpleOffset(h_i))
  pop_e = S.Population(pse,(conn_ee,pse),(conn_ei,psi),
      (S.FakeConnection(),in_state_e))
  pop_i = S.Population(psi,(conn_ie,pse),(conn_ii,psi),
      (S.FakeConnection(),in_state_i))
  ##
  dt = 1E-2
  T = 60.0
  times = 0:dt:T 
  ntimes = length(times)
  mynetwork = S.RecurrentNetwork(dt,(pop_e,pop_i))

  ei_out = Matrix{Float64}(undef,2,ntimes)
  # initial conditions
  pse.state_now .= S.ioinv(10.0,pse)
  psi.state_now .= S.ioinv(10.0,psi)

  for (k,t) in enumerate(times) 
    ei_out[1,k] = S.iofunction(pse.state_now[1],neuron_e_and_i)
    ei_out[2,k] = S.iofunction(psi.state_now[1],neuron_e_and_i)
    # rate model with constant input  does not really depend on absolute time (first argument)
    S.dynamics_step!(mynetwork)  
  end
  @test all(isapprox.(ei_out[:,end],fpoint;atol=1E-1))

end

@testset "Spike inputs in single LIF neuron" begin
  dt = 1E-3
  Ttot = 10.0
  # One LIF neuron
  myτ = 0.2
  vth = 10.
  v_r = -5.0
  τrefr= 0.3 # refractoriness
  τpcd = 0.2 # post synaptic current decay
  myinput = 0.0 # constant input to E neuron
  ps_e = S.PSLIF(myτ,vth,v_r,τrefr,τpcd,1)

  # one static input 
  in_state_e = S.PSSimpleInput(S.InputSimpleOffset(myinput))
  # connection will be FakeConnection()

  # let's produce a couple of trains
  train1 = let rat = 1.0
    sort(rand(Uniform(0.05,Ttot),round(Integer,rat*Ttot) ))
  end
  train2 = let rat = 0.5
    sort(rand(Uniform(0.05,Ttot),round(Integer,rat*Ttot) ))
  end
  # input population
  ps_train_in=S.PSFixedSpiketrain([train1,train2],myτ)

  # and connection object
  conn_e_in = let w_intrain2e = sparse([eps() Inf ; ])
    S.ConnSpikeTransfer(w_intrain2e)
  end

  # connected populations
  # two populations: the input population (unconnected) 
  # and the E neuron connected to input
  pop_in = S.UnconnectedPopulation(ps_train_in)
  pop_e = S.Population(ps_e,(conn_e_in,ps_train_in),
    (S.FakeConnection(),in_state_e))

  # that's it, let's make the network
  myntw = S.RecurrentNetwork(dt,pop_in,pop_e)

  # record spiketimes and internal potential
  krec = 1
  rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)
  rec_spikes_e = S.RecSpikes(ps_e,100.0,Ttot)
  rec_spikes_in = S.RecSpikes(ps_train_in,100.0,Ttot)

  ## Run

  times = (0:myntw.dt:Ttot)
  nt = length(times)
  # clean up
  S.reset!.([rec_state_e,rec_spikes_e,rec_spikes_in])
  S.reset!.([ps_e,ps_train_in])
  # initial conditions
  ps_e.state_now[1] = 0.0

  for (k,t) in enumerate(times)
    rec_state_e(t,k,myntw)
    rec_spikes_e(t,k,myntw)
    rec_spikes_in(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
  # this is useful for visualization only
  S.add_fake_spikes!(1.5vth,rec_state_e,rec_spikes_e)
  ##
  train1_sim,train2_sim = let (spkt,spkneu) = S.get_spiketimes_spikeneurons(rec_spikes_in)
    spkt[spkneu .== 1],spkt[spkneu .== 2]
  end
  train_e_sim = let (spkt,spkneu) = S.get_spiketimes_spikeneurons(rec_spikes_e)
    spkt
  end
  @test all(isapprox.(train1,train1_sim;atol=1.1dt))
  @test all(isapprox.(train2,train2_sim;atol=1.1dt))
  @test all(isapprox.(train_e_sim,sort(vcat(train1_sim,train2_sim));atol=1.1dt))
  # but the error between train_e_sim and train1, train2 is up to 2dt!
  ## part 2, expected period of LIF neuron
  dt = 5E-4
  Ttot = 3.0 
  myτ = 0.1
  vth = 12.
  v_r = -6.123
  τrefr = 0.0
  τpcd = 1E10
  myinput = 14.0
  ps_e = S.PSLIF(myτ,vth,v_r,τrefr,τpcd,1)
  # create static input 
  in_state_e = S.PSSimpleInput(S.InputSimpleOffset(myinput))
  # only one population: E with input
  pop_e = S.Population(ps_e,(S.FakeConnection(),in_state_e))
  # that's it, let's make the network
  myntw = S.RecurrentNetwork(dt,pop_e)

  times = (0:myntw.dt:Ttot)
  nt = length(times)
  S.expected_period_norefr(ps_e.neurontype,myinput)
  # spike recorder
  rec_spikes = let exp_freq = inv(S.expected_period_norefr(ps_e.neurontype,myinput))
    S.RecSpikes(ps_e,1.5*exp_freq,Ttot)
  end
  # reset and run 
  S.reset!(rec_spikes)
  S.reset!(ps_e)
  # initial conditions
  ps_e.state_now[1] = v_r

  # run!
  for (k,t) in enumerate(times)
    rec_spikes(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
  ##
  spkt,_ = S.get_spiketimes_spikeneurons(rec_spikes)
  # period of first spike
  @test isapprox(S.expected_period_norefr(ps_e.neurontype,myinput),spkt[1] ;
    atol = 0.02)
end


@testset "Exact spike time generation" begin
  therate = 123.4
  sgentest = S.SGPoisson(therate)

  t_final = let t_current = 0.0
    for i in 1:1_000
      t_current=S._get_spiketime_update(t_current,sgentest,2)
    end
    t_current
  end
  ratenum = 1_000/t_final
  @test(isapprox(ratenum,therate;atol=5.0))
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