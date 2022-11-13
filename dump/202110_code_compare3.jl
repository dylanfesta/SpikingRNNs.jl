#=
Here I show the exact correspondence
my spiking implementation and code by
Litwin-Kumar / Auguste Schulz
Checking I<-I and  I<-E connections 
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
function runsimulation_static_simplified(Ne::Integer,Ni::Integer,
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
		taurefrac = 10 #absolute refractory period ms
    
		# total number of neurons
		Ncells = Ne+Ni

    # constant input
    in_const = input_constant

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

## two I neurons
const Ttot=1.0

v_start = let n=2,
  (vmin,vmax) = (-60.0,-50.0)
  rand(Uniform(vmin,vmax),n)
end 


vtest_ii = let dt=0.1,
  _w = 231.0,
  input = [(70.0 - 50.0) ,0.0],
  T=round(Integer,Ttot*1E3),
  Ne=0,Ni=2,weights=[0.0 _w ;0.0 0.0];
  spiketimes=Matrix{Int32}(undef,1000,2)
  runsimulation_static_simplified(Ne,Ni,weights,spiketimes,
  v_start, input; dt=dt, T=T)[2]
end

_ = let plt = plot()
  ts = (1:size(vtest_ii,2)).*0.1E-3
  plot!(plt,ts,vtest_ii[1,:];linewidth=2)
  plot!(plt,ts,vtest_ii[2,:];linewidth=2)
end

##
vtest_ie = let dt=0.1,
  _w = 300.0,
  input = [(60.0 - 50.0) , 0.0],
  T=round(Integer,Ttot*1E3),
  Ne=1,Ni=1,weights=[0.0 _w ;0.0 0.0];
  spiketimes=Matrix{Int32}(undef,1000,2)
  runsimulation_static_simplified(Ne,Ni,weights,spiketimes,
  v_start, input; dt=dt, T=T)[2]
end

_ = let plt = plot()
  ts = (1:size(vtest_ie,2)).*0.1E-3
  plot!(plt,ts,vtest_ie[1,:];linewidth=2)
  plot!(plt,ts,vtest_ie[2,:];linewidth=2)
end

## 
# time in seconds, voltage in mV
dt = 0.1E-3
myτe = 30E-3 # seconds
myτi = 20E-3 # seconds
vreste = -60.0 
vresti = -70.0 
τrefr= 10E-3 # refractoriness
vth_e = 20.   # mV
vthexp = -52.0 # actual threshold for spike-generation
vth_i = vthexp
eifslope = 2.0
v_reset_e = vreste
v_reset_i = vresti
v_rev_e = 0.0
v_rev_i = -75.0
v_leak_e = vreste
v_leak_i = vresti
Cap = 300.0 #capacitance mF

in_const_e = (-50.0 - vreste)*Cap/myτe
in_const_i = (-50.0 - vresti)*Cap/myτi


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
   vth_e,v_reset_e,vreste,τrefr,v_rev_e)
end
nt_i = let sker = S.SKExpDiff(tauiplus,tauiminus)
  sgen = S.SpikeGenNone()
  S.NTLIFConductance(sker,sgen,myτi,Cap,
   vth_i,v_reset_i,vresti,τrefr,v_rev_i)
end
ps_e = S.PSLIFConductance(nt_e,1)
ps_i1 = S.PSLIFConductance(nt_i,1)
ps_i2 = S.PSLIFConductance(nt_i,1)

# one static input 
ps_input1e = S.PSSimpleInput(S.InputSimpleOffset(in_const_e))
ps_input1i = S.PSSimpleInput(S.InputSimpleOffset(in_const_i))

# I to E connection
conn_ii = let  _w = 231.0
  wmat = sparse(fill(_w,(1,1)))
  S.ConnGeneralIF2(wmat)
end

pop_i1 = S.Population(ps_i1,(S.InputDummyConnection(),ps_input1i))
pop_i2 = S.Population(ps_i2,(conn_ii,ps_i1))


##
# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop_i1,pop_i2)

# record spiketimes and internal potential
krec = 1
rec_state_i1 = S.RecStateNow(ps_i1,krec,dt,Ttot)
rec_state_i2 = S.RecStateNow(ps_i2,krec,dt,Ttot)
rec_spikes_i1 = S.RecSpikes(ps_i1,100.0,Ttot)
rec_spikes_i2 = S.RecSpikes(ps_i2,100.0,Ttot)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_i1,rec_spikes_i1])
S.reset!.([rec_state_i2,rec_spikes_i2])
S.reset!.([ps_e,ps_i1,ps_i2])
S.reset!(conn_ii)
# initial conditions
ps_i1.state_now .= v_start[1]
ps_i2.state_now .= v_start[2]

@time begin
  for (k,t) in enumerate(times)
    rec_state_i1(t,k,myntw)
    rec_state_i2(t,k,myntw)
    rec_spikes_i1(t,k,myntw)
    rec_spikes_i2(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
end

S.add_fake_spikes!(1.0vth_i,rec_state_i1,rec_spikes_i1)
S.add_fake_spikes!(1.0vth_i,rec_state_i2,rec_spikes_i2)
##

_ = let plt=plot()
  plot!(plt,rec_state_i1.times,rec_state_i1.state_now[1,:];linewidth=2,leg=false)
  ts = (1:size(vtest_ii,2)).*0.1E-3
  plot!(plt,ts,vtest_ii[1,:];linewidth=2,linestyle=:dash)
end

_ = let plt=plot()
  plot!(plt,rec_state_i2.times,rec_state_i2.state_now[1,:];linewidth=2,leg=false)
  ts = (1:size(vtest_ii,2)).*0.1E-3
  plot!(plt,ts,vtest_ii[2,:];linewidth=2,linestyle=:dash)
end


##

# I to E connection
conn_ie = let  _w = 300.0
  wmat = sparse(fill(_w,(1,1)))
  S.ConnGeneralIF2(wmat)
end

ps_e = S.PSLIFConductance(nt_e,1)
ps_i = ps_i1


pop_e = S.Population(ps_e,(S.InputDummyConnection(),ps_input1e))
pop_i = S.Population(ps_i,(conn_ie,ps_e))

myntw = S.RecurrentNetwork(dt,pop_e,pop_i)

krec = 1
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot)
rec_state_i = S.RecStateNow(ps_i,krec,dt,Ttot)
rec_spikes_e = S.RecSpikes(ps_e,100.0,Ttot)
rec_spikes_i = S.RecSpikes(ps_i,100.0,Ttot)

##
times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e])
S.reset!.([rec_state_i,rec_spikes_i])
S.reset!.([ps_e,ps_i])
S.reset!(conn_ie)
# initial conditions
ps_e.state_now .= v_start[1]
ps_i.state_now .= v_start[2]

@time begin
  for (k,t) in enumerate(times)
    rec_state_e(t,k,myntw)
    rec_state_i(t,k,myntw)
    rec_spikes_e(t,k,myntw)
    rec_spikes_i(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
end

S.add_fake_spikes!(1.0vth_i,rec_state_i1,rec_spikes_i1)
S.add_fake_spikes!(1.0vth_i,rec_state_i2,rec_spikes_i2)

##
_ = let plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[1,:];linewidth=2,leg=false)
  ts = (1:size(vtest_ie,2)).*0.1E-3
  plot!(plt,ts,vtest_ie[1,:];linewidth=2,linestyle=:dash)
end

_ = let plt=plot()
  plot!(plt,rec_state_i.times,rec_state_i.state_now[1,:];linewidth=2,leg=false)
  ts = (1:size(vtest_ie,2)).*0.1E-3
  plot!(plt,ts,vtest_ie[2,:];linewidth=2,linestyle=:dash)
end