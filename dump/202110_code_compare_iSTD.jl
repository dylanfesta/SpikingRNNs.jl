#=
Here I show the exact correspondence
my spiking implementation and code by
Litwin-Kumar / Auguste Schulz
When I to E plasticity is present, following Vogel's rule
=#

push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using BenchmarkTools
using ProgressMeter

function onesparsemat(w::Real)
  return sparse(fill(w,(1,1)))
end

##

using Random; Random.seed!(0)

function initweights(Ne, Ni, p, Jee0, Jei0, Jie, Jii)
	""" initialise the connectivity weight matrix based on initial
	E-to-E Jee0 initial weight e to e plastic
	I-to-E Jei0 initial weight i to e plastic
	E-to-I Jie0 constant weight e to i not plastic
	I-to-I Jii constant weight i to i not plastic"""
	Ncells = Ne+Ni

	# initialise weight matrix
	# w[i,j] is the weight from pre- i to postsynaptic j neuron
	weights = zeros(Float64,Ncells,Ncells)
	weights[1:Ne,1:Ne] .= Jee0
	weights[1:Ne,(1+Ne):Ncells] .= Jie
	weights[(1+Ne):Ncells,1:Ne] .= Jei0
	weights[(1+Ne):Ncells,(1+Ne):Ncells] .= Jii
	# set diagonal elements to 0
	# for cc = 1:Ncells
	# 	weights[cc,cc] = 0
	# end
	weights[diagind(weights)] .= 0.0
	# ensure that the connection probability is only p
	weights = weights.*(rand(Ncells,Ncells) .< p)
	return weights
end
function weightpars(;Ne = 4000, Ni = 1000, p = 0.2 )
	"""Ne, Ni number of excitatory, inhibitory neurons
	p initial connection probability"""
	Jee0 = 2.86 #initial weight e to e plastic
	Jei0 = 48.7 #initial weight i to e plastic
	Jie = 1.27 #constant weight e to i not plastic
	Jii = 16.2 #constant weight i to i not plastic
	return Ne,Ni,p,Jee0,Jei0,Jie,Jii
end
initweights() = initweights(weightpars()...)


# this is a bare-bone version of the Schulz code
function runsimulation_static_simplified_iSTD(Ne::Integer,Ni::Integer,
  weights::Array{Float64,2},spiketimes::Array{<:Integer,2},
  v_start::Vector{Float64},input_constant::Vector{Float64}; dt = 0.1, T = 2000)

		# Naming convention
		# e corresponds to excitatory
		# i corresponds to inhibitory
		# x corresponds to external

		#membrane dynamics
		taue = 30 #e membrane time constant ms
		taui = 20 #i membrane time constant ms
		vreste = -60 #e resting potential mV
		vresti = -70 #i resting potential mV
		vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset mV
		eifslope = 2 #eif slope parameter mV
		C = 300 #capacitance pF
		erev = 0 #e synapse reversal potential mV
		irev = -75 #i synapse reversal potntial mV
		vth0 = -52 #initial spike voltage threshold mV
		thrchange = false # can be switched off to have vth constant at vth0
		ath = 10 #increase in threshold post spike mV
		tauth = 30 #threshold decay timescale ms
		vresete = vreste #reset potential mV
    vreseti = vresti
		taurefrac = 15 #absolute refractory period ms
    
		# total number of neurons
		Ncells = Ne+Ni

    # constant input
    in_const = input_constant

		# synaptic kernel
		tauerise = 1 #e synapse rise time
		tauedecay = 6 #e synapse decay time
		tauirise = .5 #i synapse rise time
		tauidecay = 2 #i synapse decay time


		#inhibitory stdp
    stdpdelay = 0.0 #1000 #time before stdp is activated, to allow transients to die out ms
    ifiSTDP=true
		tauy = 20 #width of istdp curve ms
		eta = 1 #istdp learning rate pA
		r0 = .003 #target rate (khz)

		# Initialisation of spike arrays ------------------------------
		# spike count and spike times
		# Scope problems - changing arrays are therefore defined here directly

		totalspikes = zeros(Int,Ncells)
		totsp::Int64 = 0; # total numberof spikes
		spmax::Int64 = size(spiketimes,1); # maximum recorded spikes

		# further arrays for storing inputs and synaptic integration
		forwardInputsE = zeros(Float64,Ncells) #sum of all incoming weights from excitatpry inputs both external and EE and IE
		forwardInputsI = zeros(Float64,Ncells) #sum of all incoming weights from inhibitory inputs both II and EI
		forwardInputsEPrev = zeros(Float64,Ncells) #as above, for previous timestep
		forwardInputsIPrev = zeros(Float64,Ncells)

		xerise = zeros(Float64,Ncells) #auxiliary variables for E/I currents (difference of exponentials)
		xedecay = zeros(Float64,Ncells)
		xirise = zeros(Float64,Ncells)
		xidecay = zeros(Float64,Ncells)


		v = zeros(Float64,Ncells) #membrane voltage

		# initialisation of membrane potentials and poisson inputs
    copy!(v,v_start)
		# for cc = 1:Ncells
		# 	v[cc] = vreset # + (vth0-vreset)*rand()
		# end


		vth = vth0*ones(Float64,Ncells) #adaptive threshold
		lastSpike = -100*ones(Ncells) #last time the neuron spiked

		# ---------------------------------- set up storing avg weights -----------------

		# total number of simulation steps, steps when to normalise, and save
		Nsteps = round(Int,T/dt)

		# true time
		t::Float64 = 0.0

		# bool counter if a neuron has just had a spike
		spiked = zeros(Bool,Ncells)
  v_out = Matrix{Float64}(undef,Ncells,Nsteps)
	@time	for tt = 1:Nsteps

				if mod(tt,Nsteps/100) == 1  #print percent complete
					print("\r",round(Int,100*tt/Nsteps))
				end

				forwardInputsE[:] .= 0.
				forwardInputsI[:] .= 0.
				t = dt*tt

				fill!(spiked,zero(Bool)) # reset spike bool without new memory allocation

				for cc = 1:Ncells
					xerise[cc] += -dt*xerise[cc]/tauerise + forwardInputsEPrev[cc]
					xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardInputsEPrev[cc]
					xirise[cc] += -dt*xirise[cc]/tauirise + forwardInputsIPrev[cc]
					xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardInputsIPrev[cc]

					if cc <= Ne # excitatory
						if thrchange
						vth[cc] += dt*(vth0 - vth[cc])/tauth;
						end
					end

					if t > (lastSpike[cc] + taurefrac) #not in refractory period
						# update membrane voltage

						ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise);
						gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise);

						if cc <= Ne #excitatory neuron (eif), has adaptation
							dv = (vreste - v[cc] + eifslope*exp((v[cc]-vth[cc])/eifslope))/taue + 
                  ge*(erev-v[cc])/C + gi*(irev-v[cc])/C +
                  in_const[cc]/taue
							v[cc] += dt*dv
							if v[cc] > vpeak
								spiked[cc] = true
							end
						else
							dv = (vresti - v[cc])/taui +
                 ge*(erev-v[cc])/C + gi*(irev-v[cc])/C +
                  in_const[cc]/taui
							v[cc] += dt*dv
							if v[cc] > vth0
								spiked[cc] = true
							end
						end

						if spiked[cc] #spike occurred
							spiked[cc] = true;
							v[cc] = cc<=Ne ? vresete : vreseti;
							lastSpike[cc] = t;
							totalspikes[cc] += 1;
							totsp += 1;
							if totsp < spmax
								spiketimes[totsp,1] = tt; # time index as a sparse way to save spiketimes
								spiketimes[totsp,2] = cc; # cell id
							elseif totsp == spmax
								spiketimes[totsp,1] = tt; # time index
								spiketimes[totsp,2] = cc; # cell id

								totsp = 0 # reset counter total number of spikes
							end

							trace_istdp[cc] += 1.

							if cc <= Ne && thrchange # only change for excitatory cells and when thrchange == true
								vth[cc] = vth0 + ath;
							end

							#loop over synaptic projections
							for dd = 1:Ncells # postsynaptic cells dd  - cc presynaptic cells
								if cc <= Ne #excitatory synapse
									forwardInputsE[dd] += weights[cc,dd];
								else #inhibitory synapse
									forwardInputsI[dd] += weights[cc,dd];
								end
							end

						end #end if(spiked)
					end #end if(not refractory)

          if ifiSTDP # select if iSTDP
						#istdp
						if spiked[cc] && (t > stdpdelay)
							if cc < Ne #excitatory neuron fired, potentiate i inputs
								for dd = (Ne+1):Ncells
									if weights[dd,cc] == 0.
										continue
									end
									weights[dd,cc] += eta*trace_istdp[dd]

									if weights[dd,cc] > Jeimax
										weights[dd,cc] = Jeimax
									end
								end
							else #inhibitory neuron fired, modify outputs to e neurons
								for dd = 1:Ne
									if weights[cc,dd] == 0.
										continue
									end

									weights[cc,dd] += eta*(trace_istdp[dd] - 2*r0*tauy)
									if weights[cc,dd] > Jeimax
										weights[cc,dd] = Jeimax
									elseif weights[cc,dd] < Jeimin
										weights[cc,dd] = Jeimin
									end
								end
							end
						end #end istdp
					end # ifiSTDP

        v_out[cc,tt] = v[cc]
			end #end loop over cells

			forwardInputsEPrev = copy(forwardInputsE)
			forwardInputsIPrev = copy(forwardInputsI)

		end # tt loop over time
	print("\r")

	println("simulation finished")

	return totalspikes,v_out

end 

##

const Ne = 4000
const Ni = 1000
const Ttot = 20.0
const all_in_e = (60.0-52.5)
const all_in_i = (70.0-53.0)


weights = permutedims(initweights(weightpars(;Ne=Ne,Ni=Ni)...))

v_start = let n=Ne+Ni,
  (vmin,vmax) = (-60.0,-51.0)
  rand(Uniform(vmin,vmax),n)
end 
const n_spikes_count  = 100_000
test_spiketimes = fill(-1,(n_spikes_count,2))

totspikes,vtest = let dt=0.1,
  _weights = permutedims(weights),
  input = vcat(fill(all_in_e,Ne),fill(all_in_i,Ni)),
  T=round(Integer,Ttot*1E3)
  runsimulation_static_simplified(Ne,Ni,_weights,test_spiketimes,v_start,
    input;dt=dt, T=T)
end
rates_test = totspikes ./ Ttot

_ = let plt = plot(),nneu=22,
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[nneu,:];linewidth=2,xlims=(0,1))
end


histogram(rates_test[1:Ne])

##
#=

		#membrane dynamics
		taue = 30 #e membrane time constant ms
		taui = 20 #i membrane time constant ms
		vreste = -60 #e resting potential mV
		vresti = -70 #i resting potential mV
		vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset mV
		eifslope = 2 #eif slope parameter mV
		C = 300 #capacitance pF
		erev = 0 #e synapse reversal potential mV
		irev = -75 #i synapse reversal potntial mV
		vth0 = -52 #initial spike voltage threshold mV
		thrchange = false # can be switched off to have vth constant at vth0
		ath = 10 #increase in threshold post spike mV
		tauth = 30 #threshold decay timescale ms
		vreset = -70 #reset potential mV
		taurefrac = 50 #absolute refractory period ms
		aw_adapt = 4 #adaptation parameter a nS conductance
		bwfactor = 100
		bw_adapt = bwfactor*0.805 #adaptation parameter b pA current

    # constant input
    in_const = fill(0.0,Ncells)
    in_const[2] = -51.0 - vreste

		# synaptic kernel
		tauerise = 1 #e synapse rise time
		tauedecay = 6 #e synapse decay time
		tauirise = .5 #i synapse rise time
		tauidecay = 2 #i synapse decay time

=#

## 
# time in seconds, voltage in mV
const dt = 0.1E-3
myτe = 30E-3 # seconds
myτi = 20E-3 # seconds
τrefr= 15E-3 # refractoriness
vth_e = 20.   # mV
vthexp = -52.0 # actual threshold for spike-generation
vth_i = vthexp
eifslope = 2.0
v_rest_e = -60.0
v_rest_i = -70.0
v_rev_e = 0.0
v_rev_i = -75.0
v_leak_e = v_rest_e
v_leak_i = v_resti
v_reset_e = v_rest_e
v_reset_i = v_rest_i
Cap = 300.0 #capacitance mF
in_const_e = all_in_e*Cap/myτe
in_const_i = all_in_i*Cap/myτi


# synaptic kernel
tauerise = 1E-3 #e synapse rise time
tauedecay = 6E-3 #e synapse decay time
taueplus,taueminus  = tauedecay, tauerise
tauirise = 0.5E-3 #i synapse rise time
tauidecay = 2E-3 #i synapse decay time
tauiplus,tauiminus = tauidecay,tauirise

##


nt_e = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SpikeGenEIF(vthexp,eifslope)
  S.NTLIFConductance(sker,sgen,myτe,Cap,
   vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
end
ps_e = S.PSLIFConductance(nt_e,Ne)

nt_i = let sker = S.SKExpDiff(tauiplus,tauiminus)
  sgen = S.SpikeGenNone()
  S.NTLIFConductance(sker,sgen,myτi,Cap,
   vth_i,v_reset_i,v_rest_i,τrefr,v_rev_i)
end
ps_i = S.PSLIFConductance(nt_i,Ni)

# static inputs
ps_input_e = S.PSSimpleInput(S.InputSimpleOffset(in_const_e))
ps_input_i = S.PSSimpleInput(S.InputSimpleOffset(in_const_i))

## connections 
conn_ee = let w_ee = weights[1:Ne,1:Ne]
  S.ConnGeneralIF2(sparse(w_ee))
end
conn_ii = let w_ii = weights[Ne+1:end,Ne+1:end]
  S.ConnGeneralIF2(sparse(w_ii))
end
conn_ei = let w_ei = weights[1:Ne,Ne+1:end]
  S.ConnGeneralIF2(sparse(w_ei))
end
conn_ie = let w_ie = weights[Ne+1:end,1:Ne]
  S.ConnGeneralIF2(sparse(w_ie))
end

# input populations
pop_e = S.Population(ps_e,(S.FakeConnection(),ps_input_e),
   (conn_ee,ps_e),(conn_ei,ps_i) )
pop_i = S.Population(ps_i,(S.FakeConnection(),ps_input_i),
   (conn_ii,ps_i),(conn_ie,ps_e) )

##
# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop_e,pop_i)

# record spiketimes and internal potential
krec = 1
n_e_rec = 1000
n_i_rec = 1000
t_wup = 0.0
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_state_i = S.RecStateNow(ps_i,krec,dt,Ttot;idx_save=collect(1:n_i_rec),t_warmup=t_wup)
rec_spikes_e = S.RecSpikes(ps_e,5.0,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_spikes_i = S.RecSpikes(ps_i,5.0,Ttot;idx_save=collect(1:n_i_rec),t_warmup=t_wup)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e])
S.reset!.([rec_state_i,rec_spikes_i])
S.reset!.([ps_e,ps_i])
S.reset!(conn_ei)
# initial conditions
ps_e.state_now .= v_start[1:Ne]
ps_i.state_now .= v_start[Ne+1:end]

@time begin
  @showprogress 5.0 "network simulation " for (k,t) in enumerate(times)
    rec_state_e(t,k,myntw)
    rec_state_i(t,k,myntw)
    rec_spikes_e(t,k,myntw)
    rec_spikes_i(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
end

#S.add_fake_spikes!(1.0vth_e,rec_state_e,rec_spikes_e)
#S.add_fake_spikes!(0.0,rec_state_i,rec_spikes_i)
##

rates_e = let rdic=S.get_mean_rates(rec_spikes_e,dt,Ttot)
  ret = fill(0.0,n_i_rec)
  for (k,v) in pairs(rdic)
    ret[k] = v
  end
  ret
end
rates_i = let rdic=S.get_mean_rates(rec_spikes_i,dt,Ttot)
  ret = fill(0.0,n_i_rec)
  for (k,v) in pairs(rdic)
    ret[k] = v
  end
  ret
end


_ = let plt=plot(;leg=false),
  netest = n_e_rec
  scatter!(rates_e,rates_test[1:netest];ratio=1,
    xlabel="Dylan's model",ylabel="Suchulz et al")
  plot!(plt,identity; linewidth=2)
end

_ = let plt=plot(;leg=false),
  ntest = n_i_rec
  scatter!(rates_i,rates_test[Ne+1:Ne+ntest];ratio=1,
    xlabel="Dylan's model",ylabel="Suchulz et al")
  plot!(plt,identity;linewidth=2)
end

##


_ = let neu = 202,
  plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[neu,:];linewidth=2,leg=false,
    xlims=(0,1),ylims=(-70,-40))
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[neu,:];linewidth=2,linestyle=:dash)
end
_ = let plt=plot(),neu=16
  plot!(plt,rec_state_i.times,rec_state_i.state_now[neu,:];linewidth=2,leg=false,
    xlims=(0,1))
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[Ne+neu,:];linewidth=2,linestyle=:dash)
end
