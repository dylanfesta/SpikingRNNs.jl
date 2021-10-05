

function runsimulation_static_simplified(Ne::Integer,Ni::Integer,
  weights::Array{Float64,2},spiketimes::Array{Int32,2}; dt = 0.1, T = 2000)

		# Naming convention
		# e corresponds to excitatory
		# i corresponds to inhibitory
		# x corresponds to external

		#membrane dynamics
		taue = 20 #e membrane time constant ms
		taui = 20 #i membrane time constant ms
		vreste = -70 #e resting potential mV
		vresti = -62 #i resting potential mV
		vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset mV
		eifslope = 2 #eif slope parameter mV
		C = 300 #capacitance pF
		erev = 0 #e synapse reversal potential mV
		irev = -75 #i synapse reversal potntial mV
		vth0 = -52 #initial spike voltage threshold mV
		thrchange = false # can be switched off to have vth constant at vth0
		ath = 10 #increase in threshold post spike mV
		tauth = 30 #threshold decay timescale ms
		vreset = -60 #reset potential mV
		taurefrac = 50 #absolute refractory period ms

    # constant input
    in_const_e = -51.0 - vreste
    in_const_i = -51.0 - vresti


		# total number of neurons
		Ncells = Ne+Ni

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
                  in_const_e/taue
							v[cc] += dt*dv
							if v[cc] > vpeak
								spiked[cc] = true
							end
						else
							dv = (vresti - v[cc])/taui +
                 ge*(erev-v[cc])/C + gi*(irev-v[cc])/C
                 +  in_const_i/taui
							v[cc] += dt*dv;
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



# second version, with different input on each neuron


function runsimulation_static_simplified2(Ne::Integer,Ni::Integer,
  weights::Array{Float64,2},spiketimes::Array{Int32,2}; dt = 0.1, T = 2000)

		# Naming convention
		# e corresponds to excitatory
		# i corresponds to inhibitory
		# x corresponds to external

		#membrane dynamics
		taue = 20 #e membrane time constant ms
		taui = 20 #i membrane time constant ms
		vreste = -60 #e resting potential mV
		vresti = -60 #i resting potential mV
		vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset mV
		eifslope = 2 #eif slope parameter mV
		C = 300 #capacitance pF
		erev = 0 #e synapse reversal potential mV
		irev = -75 #i synapse reversal potntial mV
		vth0 = -52 #initial spike voltage threshold mV
		thrchange = false # can be switched off to have vth constant at vth0
		ath = 10 #increase in threshold post spike mV
		tauth = 30 #threshold decay timescale ms
		vreset = -60 #reset potential mV
		taurefrac = 50 #absolute refractory period ms
    
		# total number of neurons
		Ncells = Ne+Ni

    # constant input
    in_const = fill(0.0,Ncells)
    in_const[1] = -51.0 - vreste

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
                 ge*(erev-v[cc])/C + gi*(irev-v[cc])/C
                 +  in_const[cc]/taui
							v[cc] += dt*dv;
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
