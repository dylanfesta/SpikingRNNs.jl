#=
Here I show the exact correspondence
my spiking implementation and code by
Litwin-Kumar / Auguste Schulz
For one I neuron connected to one E neuron
=#

push!(LOAD_PATH, abspath(@__DIR__,".."))

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark)
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs

function onesparsemat(w::Real)
  return sparse(fill(w,(1,1)))
end

##

# this is a bare-bone version of the Schulz code
function runsimulation_static_simplified2(Ne::Integer,Ni::Integer,
  weights::Array{Float64,2},spiketimes::Array{Int32,2}; dt = 0.1, T = 2000)

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
		vreset = -70 #reset potential mV
		taurefrac = 50 #absolute refractory period ms
    
		# total number of neurons
		Ncells = Ne+Ni

    # constant input
    in_const = fill(0.0,Ncells)
    in_const[2] = -51.0 - vresti

		# synaptic kernel
		tauerise = 1 #e synapse rise time
		tauedecay = 6 #e synapse decay time
		tauirise = .5 #i synapse rise time
		tauidecay = 2 #i synapse decay time


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
		for cc = 1:Ncells
			v[cc] = vreset # + (vth0-vreset)*rand()
		end


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
							v[cc] = vreset;
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
vtest = let dt=0.1,T=500,
  _w = 123.0,
  Ne=1,Ni=1,weights=[0.0 0.0 ; _w 0.0];
  spiketimes=Matrix{Int32}(undef,1000,1000)
  runsimulation_static_simplified2(Ne,Ni,weights,spiketimes;
   dt=dt, T=T)[2]
end

_ = let plt = plot()
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[1,:];linewidth=2)
  plot!(plt,ts,vtest[2,:];linewidth=2,linestyle=:dash)
end

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
dt = 0.1E-3
Ttot = 0.5
myτe = 30E-3 # seconds
myτi = 20E-3 # seconds
τrefr= 50E-3 # refractoriness
vth = 20.   # mV
vthexp = -52.0 # actual threshold for spike-generation
vth_i = vthexp
eifslope = 2.0
v_reset = -70.0
v_rev_e = 0.0
v_rev_i = -75.0
vreste = -60.0 #e resting potential mV
v_leak_e = vreste
vresti = -70.0 #i resting potential mV
in_const_e = 0.0 
Cap = 300.0 #capacitance mF
in_const_i = (-51.0 - vresti)*Cap/myτi


# synaptic kernel
tauerise = 1E-3 #e synapse rise time
tauedecay = 6E-3 #e synapse decay time
#taueplus,taueminus  = tauerise, tauedecay
taueplus,taueminus  = tauedecay, tauerise
tauirise = 0.5E-3 #i synapse rise time
tauidecay = 2E-3 #i synapse decay time
tauiplus,tauiminus = tauidecay,tauirise

##
nt_e = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SpikeGenEIF(vthexp,eifslope)
  S.NTLIFConductance(sker,sgen,myτe,Cap,
   vth,v_reset,vreste,τrefr,v_rev_e)
end
nt_i = let sker = S.SKExpDiff(tauiplus,tauiminus)
  sgen = S.SpikeGenNone()
  S.NTLIFConductance(sker,sgen,myτi,Cap,
   vth_i,v_reset,vresti,τrefr,v_rev_i)
end
ps_e = S.PSLIFConductance(nt_e,1)
ps_i = S.PSLIFConductance(nt_i,1)

# one static input 
ps_input1 = S.PSSimpleInput(S.InputSimpleOffset(in_const_i))

# I to E connection
conn_ei = let  _w = 123.0
  wmat = sparse(fill(_w,(1,1)))
  S.ConnGeneralIF2(wmat)
end

pop_i = S.Population(ps_i,(S.InputDummyConnection(),ps_input1))
pop_e = S.Population(ps_e,(conn_ei,ps_i))


##
# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop_e,pop_i)

# record spiketimes and internal potential
krec = 1
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)
rec_state_i = S.RecStateNow(ps_i,krec,dt,Ttot)
rec_spikes_e = S.RecSpikes(ps_e,100.0,Ttot)
rec_spikes_i = S.RecSpikes(ps_i,100.0,Ttot)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e])
S.reset!.([rec_state_i,rec_spikes_i])
S.reset!.([ps_e,ps_i])
S.reset!(conn_ei)
# initial conditions
ps_i.state_now .= v_reset
ps_e.state_now .= v_reset

for (k,t) in enumerate(times)
  rec_state_e(t,k,myntw)
  rec_state_i(t,k,myntw)
  rec_spikes_e(t,k,myntw)
  rec_spikes_i(t,k,myntw)
  S.dynamics_step!(t,myntw)
end

S.add_fake_spikes!(1.0vth,rec_state_e,rec_spikes_e)
S.add_fake_spikes!(1.0vth_i,rec_state_i,rec_spikes_i)
##
_ = let plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[1,:];linewidth=2,leg=false)
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[1,:];linewidth=2,linestyle=:dash)
end


##

_ = let plt = plot()
  plot!(plt,rec_state_i.times,rec_state_i.state_now[1,:];linewidth=2,leg=false)
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[2,:];linewidth=2,linestyle=:dash)
end
