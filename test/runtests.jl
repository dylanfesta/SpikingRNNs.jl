using SpikingRNNs ; global const S=SpikingRNNs
using Test
using LinearAlgebra,SparseArrays,Statistics,Distributions
using Random ; Random.seed!(0)


# utility test functions

# connections
function onesparsemat(w::Real)
  return sparse(cat(w;dims=2))
end

# testing utility function
function sparse_thingies(w::SparseMatrixCSC)
  return nonzeros(w),rowvals(w),SparseArrays.getcolptr(w)
end


function wstick(wee::M,wie::M,wei::M,wii::M) where {R<:Real,M<:AbstractMatrix{R}}
  return Matrix(hcat(vcat(wee,wie), (-1).*abs.(vcat(wei,wii) )))
end

function rates_analytic(W::Matrix{R},h::Vector{R}) where R
  return (I-W)\h
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

  wtest = S.make_sparse_weights(m,n,ptest,-0.123)
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
      (S.InputDummyConnection(),in_state_e))
  pop_i = S.Population(psi,(conn_ie,pse),(conn_ii,psi),
      (S.InputDummyConnection(),in_state_i))
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
  # connection will be InputDummyConnection()

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
    (S.InputDummyConnection(),in_state_e))

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
  pop_e = S.Population(ps_e,(S.InputDummyConnection(),in_state_e))
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

@testset "Inputs faster than Euler Δt" begin
  # first test , Poissoin fixed input, faster than Euler dt
  # I expect that doubling the rate will double the firing
  # careful:
  # this would not work with an exponential spike generation kernel

  dt = 0.1E-3
  myτe = 20E-3 # seconds
  τrefr= 0.1E-3 # refractoriness
  vth_e = -20.0   # mV
  Cap = 300.0 #capacitance mF
  v_rest_e = -60.0
  v_rev_e = 0.0
  v_leak_e = v_rest_e
  v_reset_e = v_rest_e

  # synaptic kernel
  taueplus = 6E-3 #e synapse decay time
  taueminus = 1E-3 #e synapse rise time


  Ne = 200
  # if spike genrator kernel is absent, then inputs scale linearly
  nt_e = let sker = S.SKExpDiff(taueplus,taueminus)
    sgen = S.SpikeGenNone()
    S.NTLIFConductance(sker,sgen,myτe,Cap,
      vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
  end
  ps_e = S.PSLIFConductance(nt_e,Ne)


  nt_in = let in_rate = 10E3
    sker = S.SKExpDiff(taueplus,taueminus)
    sgen = S.SGPoisson(in_rate)
    S.NTInputConductance(sgen,sker,v_rev_e) 
  end
  in_weight = 7.0
  ps_in = S.PSInputPoissonConductanceExact(nt_in,in_weight,Ne)

  pop_e = S.Population(ps_e,(S.InputDummyConnection(),ps_in))
  ntw = S.RecurrentNetwork(dt,pop_e)

  Ttot = 2.0
  # record spiketimes and internal potential
  n_e_rec = Ne
  t_wup = 0.0
  rec_spikes_e = S.RecSpikes(ps_e,500.0,Ttot;idx_save=collect(1:n_e_rec),Tstart=t_wup)

  times = (0:ntw.dt:Ttot)
  nt = length(times)
  S.reset!.([ps_e,ps_in,rec_spikes_e])
  # initial conditions
  ps_e.state_now .= v_reset_e
  for (k,t) in enumerate(times)
    rec_spikes_e(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
# _ = let plt=plot(),ts=rec_state_e.times,
#   neu = 1 , vs = rec_state_e.state_now[neu,:]
#   plot!(plt,ts,vs;leg=false,linewidth=2)
# end
  rates_e = let rdic=S.get_mean_rates(rec_spikes_e)
    ret = fill(0.0,n_e_rec)
    for (k,v) in pairs(rdic)
      ret[k] = v
    end
    ret
  end
  rates_input1 = mean(rates_e)

  # now double the input rate, see if output doubles too

  nt_in = let in_rate = 2*10E3
    sker = S.SKExpDiff(taueplus,taueminus)
    sgen = S.SGPoisson(in_rate)
    S.NTInputConductance(sgen,sker,v_rev_e) 
  end
  ps_in = S.PSInputPoissonConductanceExact(nt_in,in_weight,Ne)
  pop_e = S.Population(ps_e,(S.InputDummyConnection(),ps_in))
  ntw = S.RecurrentNetwork(dt,pop_e)

  rec_spikes_e = S.RecSpikes(ps_e,500.0,Ttot;idx_save=collect(1:n_e_rec),Tstart=t_wup)

  S.reset!.([ps_e,ps_in,rec_spikes_e])
  # initial conditions
  ps_e.state_now .= v_reset_e

  for (k,t) in enumerate(times)
    rec_spikes_e(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end

  rates_e = let rdic=S.get_mean_rates(rec_spikes_e)
    ret = fill(0.0,n_e_rec)
    for (k,v) in pairs(rdic)
      ret[k] = v
    end
    ret
  end
  rates_input2 = mean(rates_e)
  @test isapprox(2*rates_input1,rates_input2;rtol=0.2)
end

@testset "Poisson network" begin

  N = 100
  dt = 0.1E-3
  τ = 50E-3
  h_in = 123.45

  # population
  ps =  S.PSPoissonNeuron(τ,N)
  # input
  ps_in = S.PoissonInputCurrentConstant(fill(h_in,N))
  # non-existing recurring connection
  conn_ee = S.ConnectionPoissonExpKernel(S.PoissonExcitatory(),-1E6,fill(0.0,N,N))

  # population
  pop = S.Population(ps,(conn_ee,ps),(S.InputDummyConnection(),ps_in))

  # network
  ntw = S.RecurrentNetwork(dt,pop)

  Ttot = 30.0
  # record spiketimes and internal potential
  rec_spikes_e = S.RecSpikes(ps,200.0,Ttot)
  rec_state_e  = S.RecStateNow(ps,1,dt,Ttot;idx_save=[1,2])

  ## Run

  times = (0:ntw.dt:Ttot)
  nt = length(times)
  # clean up
  S.reset!.([ps,rec_spikes_e])
  # initial conditions
  ps.state_now .= 30.0

  for (k,t) in enumerate(times)
    rec_spikes_e(t,k,ntw)
    rec_state_e(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end

  spikes_c = S.get_content(rec_spikes_e)
  states_c = S.get_content(rec_state_e)

  # myrast = S.draw_spike_raster(S.get_spiketrains(spikes_c)[1],0.001,0.5)
  # save("/tmp/rast.png",myrast)

  rats = S.get_mean_rates(spikes_c)

  @test all(isapprox.(values(rats),h_in;rtol=0.2))


  # test 2, rate for 500 neurons should be the similar to 2D system 
  ne = 350
  ni = 150
  dt = 0.1E-3
  τe,τi = 0.5,0.2
  he,hi = 70.,5.

  sparse_ee = 0.3
  sparse_ie = 0.3
  sparse_ei = 0.4
  sparse_ii = 0.4

  τker_e = 0.8
  τker_i = 0.4
  wee_scal,wie_scal,wei_scal,wii_scal=(1.5,2.0,1.3,.8)

  w_ee = S.sparse_constant_wmat(ne,ne,sparse_ee,1.0;rowsum=wee_scal) 
  w_ie = S.sparse_constant_wmat(ni,ne,sparse_ie,1.0;rowsum=wie_scal,no_autapses=false) 
  w_ei = S.sparse_constant_wmat(ne,ni,sparse_ei,1.0;rowsum=wei_scal,no_autapses=false) 
  w_ii = S.sparse_constant_wmat(ni,ni,sparse_ii,1.0;rowsum=wii_scal) 

  Wdense = wstick(w_ee,w_ie,w_ei,w_ii)
  h_full = vcat(fill(he,ne),fill(hi,ni))
  rats_an = rates_analytic(Wdense,h_full)
  rats_an_e = rats_an[1]
  rats_an_i = rats_an[end]

  ps_e =  S.PSPoissonNeuron(τe,ne)
  ps_i =  S.PSPoissonNeuron(τi,ni)
  conn_ee = S.ConnectionPoissonExpKernel(S.PoissonExcitatory(),τker_e,w_ee)
  conn_ie = S.ConnectionPoissonExpKernel(S.PoissonExcitatory(),τker_e,w_ie)
  conn_ei = S.ConnectionPoissonExpKernel(S.PoissonInhibitory(),τker_i,w_ei)
  conn_ii = S.ConnectionPoissonExpKernel(S.PoissonInhibitory(),τker_i,w_ii)
  # inputs
  in_e = S.PoissonInputCurrentConstant(fill(he,ne))
  in_i = S.PoissonInputCurrentConstant(fill(hi,ni))
  pop_e = S.Population(ps_e,
    (conn_ee,ps_e),(conn_ei,ps_i),(S.InputDummyConnection(),in_e))
  pop_i = S.Population(ps_i,
    (conn_ie,ps_e),(conn_ii,ps_i),(S.InputDummyConnection(),in_i))
  ntw = S.RecurrentNetwork(dt,pop_e,pop_i)
  Ttot =40.0
  # record spiketimes and internal potential
  rec_spikes_e = S.RecSpikes(ps_e,200.0,Ttot)
  rec_spikes_i = S.RecSpikes(ps_i,200.0,Ttot)

  rec_state_e  = S.RecStateNow(ps_e,10,dt,Ttot;idx_save=[1,2,3])
  rec_state_i  = S.RecStateNow(ps_i,10,dt,Ttot;idx_save=[1,2,3])

  times = (0:ntw.dt:Ttot)
  nt = length(times)
  # clean up
  S.reset!.([ps_e,ps_i,rec_spikes_e,rec_spikes_i,rec_state_e,rec_state_i])
  # initial conditions
  ps_e.state_now .= 50.0
  ps_i.state_now .= 50.0

  for (k,t) in enumerate(times)
    rec_spikes_e(t,k,ntw)
    rec_spikes_i(t,k,ntw)
    rec_state_e(t,k,ntw)
    rec_state_i(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end

  spikec_e = S.get_content(rec_spikes_e)
  spikec_i = S.get_content(rec_spikes_i)

  rats_e = collect(values(S.get_mean_rates(spikec_e;Tstart=10.0)))
  rats_i =collect(values( S.get_mean_rates(spikec_i;Tstart=10.0)))
  @test all(isapprox.(rats_e,rats_an_e;rtol=0.1))
  @test all(isapprox.(rats_i,rats_an_i;rtol=0.1))

end

#=

@testset "Homeostatic plasticity, easy implementation" begin

  # incoming connections : rows
  # Npost neurons have Npre incoming connections
  Npost = 13
  Npre = 17
  sum_max = Npre/2 - 0.1
  wtest_in_start = sparse(rand(Npost,Npre))
  sum_start = sum(wtest_in_start;dims=2)
  plast_method = S.HeterosynapticAdditive(-Inf,10.0)
  plast_target = S.HeterosynapticIncoming(sum_max)
  wtest_in = copy(wtest_in_start)
  S._apply_easy_het_plasticity!(wtest_in,plast_method,plast_target)
  sum_end = sum(wtest_in;dims=2)
  @test all(sum_end .<= sum_max*(1+1E-3))
  idx_good = findall(<=(sum_max),sum_start)
  @test all(sum_end[idx_good] .== sum_start[idx_good])
  
  # outgoing connections, columns
  
  Npost = 11
  Npre = 19
  sum_max = Npost/2 - 0.1
  wtest_in_start = sparse(rand(Npost,Npre))
  sum_start = sum(wtest_in_start;dims=2)
  plast_method = S.HeterosynapticAdditive(-Inf,10.0)
  plast_target = S.HeterosynapticOutgoing(sum_max)
  wtest_in = copy(wtest_in_start)
  S._apply_easy_het_plasticity!(wtest_in,plast_method,plast_target)
  sum_end = sum(wtest_in;dims=1)
  @test all(sum_end .<= sum_max+1E-6)
  idx_good = findall(<=(sum_max),sum_start)
  @test all(sum_end[idx_good] .== sum_start[idx_good])
  
  # Internals for other method
  N = 400
  sp = 0.4
  testmat = sprand(N,N,sp) 
  meansum = N*sp*0.5 
  maxsum = meansum - 10.0

  hup = S.HetUpperLimit(maxsum,-Inf,Inf,0.0)
  hprecise = S.HetStrictSum(maxsum,-Inf,Inf,1E-4)
  hetincoming = S.HetIncoming()
  hetoutgoing = S.HetOutgoing()
  hetboth = S.HetBoth()

  idxtry = sample((1:N),200;replace=false)
  # outgoing, along columns
  tochange,sumsvals,nels =  S._hetplast_check(idxtry,testmat,
      S.HetOutgoing(),hup)
  testmatfix = copy(testmat)
  S._apply_hetplast_spiketriggered!(tochange,idxtry,sumsvals,nels,
    testmatfix, S.HetOutgoing(),S.HetAdditive(),
    hup)
  @test all(sum(testmatfix;dims=1)[idxtry] .< (maxsum + 1E-5)) 

  # incoming, along rows
  tochange,sumsvals,nels =  S._hetplast_check(idxtry,testmat,
      S.HetIncoming(),hprecise)

  testmatfix = copy(testmat)
  S._apply_hetplast_spiketriggered!(tochange,idxtry,sumsvals,nels,
    testmatfix, S.HetIncoming(),S.HetAdditive(),
    hprecise)
  @test all(isapprox.(sum(testmatfix;dims=2)[idxtry],maxsum;atol=1E-3))

  # both, exact sum, for a single neuron
  _lowval = 1.234
  hprecise_low = S.HetStrictSum(_lowval,-Inf,Inf,1E-4)
  # idxtry = [div(N,3),]
  idxtry = [div(N,3)]

  tochange,sumsvals,nels  =  S._hetplast_check(idxtry,testmat,
      S.HetBoth(),hprecise_low)
  testmatfix = copy(testmat)
  S._apply_hetplast_spiketriggered!(tochange,idxtry,sumsvals,nels,
    testmatfix, S.HetBoth(),S.HetAdditive(),
    hprecise_low)

  @test isapprox(sum(testmatfix[:,idxtry]),_lowval;atol=1E-4)
  @test isapprox(sum(testmatfix[idxtry,:]),_lowval;atol=1E-4)
  # TO-DO  a test with both, but where only row/col passes the limit


  # more subroutines
  N = 400
  mattest = sparse(fill(1.0,N,N))
  maxsum = N/2

  constr = S.HetStrictSum(maxsum,-Inf,Inf,1E-4)

  allocrows = fill(NaN,N)
  alloccols = fill(NaN,N)

  S._het_plasticity_fix_rows!(allocrows,fill(0,N),
      mattest,constr,S.HetAdditive(),S.HetBoth())
  S._het_plasticity_fix_cols!(alloccols,fill(0,N),
      mattest,constr,S.HetAdditive(),S.HetBoth())
  # apply the fix  
  @test all( isapprox.(allocrows,-maxsum/N;atol=1E-4))
  @test all( isapprox.(alloccols,-maxsum/N;atol=1E-4))
  mattestfix = copy(mattest)
  S._het_plasticity_apply_fix!( 
        allocrows,alloccols,
        mattestfix,
        constr,S.HetAdditive(),S.HetBoth())
  
  @test all(isapprox.(sum(mattestfix;dims=1),maxsum;atol=1E-3))
  @test all(isapprox.(sum(mattestfix;dims=2),maxsum;atol=1E-3))

end

@testset "Structural plasticity" begin
  Ntot = 1000
  Ttot = 5.0
  dt = 0.1E-3
  mydensity = 0.3
  rates = rand(Uniform(20.,50.),Ntot)

  # test 1 : only death, connections expected to drop of half
  # if death freq is log(2)/Tot  (because half-life is log(2)/freq )

  # dummy neurons
  nt_all = let sker = S.SKExp(Inf)
    spkgen = S.SGPoissonMulti(rates)
    S.NTInputConductance(spkgen,sker,-Inf)
  end
  ps_all = S.PSInputConductance(nt_all,Ntot)

  ## initial weights

  weights_start = S.sparse_constant_wmat(Ntot,Ntot,mydensity,1.0;no_autapses=true)

  ## plasticity rule
  plast_struct = let  νdeath = log(2)/(Ttot),
    Δt = 30E-3,
    ρ = 0.0
    syngen = S.SynapticGenerationConstant(1.11)
    srt_type = S.StructuralPlasticityPlain(; 
      connection_density=ρ,
      death_rate = νdeath)
    S.PlasticityStructural(srt_type,syngen,Δt)
  end
  ##
  conn_all=S.ConnectionPlasticityTest(copy(weights_start),plast_struct)
  pop_all=S.Population(ps_all,(conn_all,ps_all))
  myntw = S.RecurrentNetwork(dt,pop_all)
  ## Run
  times = (0:myntw.dt:Ttot)
  nt = length(times)
  # reset weights
  copy!(conn_all.weights,weights_start)
  wstart = copy(weights_start)
  density_start = nnz(wstart)/Ntot^2
  @test isapprox(density_start,mydensity;atol=0.01)
  # clean up
  S.reset!(ps_all)
  S.reset!(conn_all)

  for (k,t) in enumerate(times)
    S.dynamics_step!(t,myntw)
  end
  wend = conn_all.weights
  density_end =  nnz(wend)/Ntot^2
  @test isapprox(density_end*2,mydensity;atol=0.03)


  # test 2 : 1/10 of connections survive, but density is preserved
  # so I use νdeath = log(10)/(Ttot)

  nt_all = let sker = S.SKExp(Inf)
    spkgen = S.SGPoissonMulti(rates)
    S.NTInputConductance(spkgen,sker,-Inf)
  end
  ps_all = S.PSInputConductance(nt_all,Ntot)

  ## initial weights
  weights_start = S.sparse_constant_wmat(Ntot,Ntot,mydensity,1.0;no_autapses=true)

  ## plasticity rule
  plast_struct = let  νdeath = log(10)/(Ttot),
    Δt = 30E-3,
    ρ = mydensity,
    syngen = S.SynapticGenerationConstant(2.0)
    srt_type = S.StructuralPlasticityPlain(; 
      connection_density=ρ,
      death_rate = νdeath)
    S.PlasticityStructural(srt_type,syngen,Δt)
  end

  conn_all=S.ConnectionPlasticityTest(copy(weights_start),plast_struct)
  pop_all=S.Population(ps_all,(conn_all,ps_all))
  myntw = S.RecurrentNetwork(dt,pop_all)

  times = (0:myntw.dt:Ttot)
  nt = length(times)
  # reset weights
  copy!(conn_all.weights,weights_start)
  wstart = copy(weights_start)
  density_start = nnz(wstart)/Ntot^2
  # clean up
  S.reset!(ps_all)
  S.reset!(conn_all)

  for (k,t) in enumerate(times)
    S.dynamics_step!(t,myntw)
  end
  wend = conn_all.weights
  density_end =  nnz(wend)/Ntot^2
  @test isapprox(density_end,density_start;atol=0.01)
  n_start = count(==(1.0),nonzeros(wstart)) 
  @test n_start == nnz(wstart)
  n_kept = count(==(1.0),nonzeros(wend)) 
  density_survivors = n_kept/Ntot^2
  @test isapprox(density_survivors*10,density_start;atol=0.02)
end

=#