# ./src/lif_current.jl

############
# Current-based (classic) LIF

# threshold-linear input-output function
struct NTLIF <: NeuronType
  τ::Float64 # time constant
	v_threshold::Float64 # fixed spiking threshold
	v_reset::Float64 # reset after spike
	τ_refractory::Float64 # refractory time
	τ_output_decay::Float64 # decay of postsynaptic currents
end

struct PSLIF{NT} <: PSGeneralCurrentIFType{NT}
  neurontype::NT
  n::Int64 # pop size
  state_now::Vector{Float64}
  input::Vector{Float64}
	alloc_dv::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
end
function PSLIF(p::NTLIF,n::Int64)
  PSLIF(p,n,ntuple(_-> zeros(Float64,n),4)...,ntuple(_-> falses(n),2)...)
end
function PSLIF(τ::Float64,v_threshold::Float64,
	  v_reset::Float64 ,
	  τ_refractory::Float64 ,
	  τ_output_decay::Float64, n::Int64)
  nt=NTLIF(τ,v_threshold,v_reset,τ_refractory,τ_output_decay)
  return PSLIF(nt,n)
end
function reset!(ps::PSLIF)
  fill!(ps.state_now,0.0)
  fill!(ps.input,0.0)
  fill!(ps.last_fired,-Inf)
  fill!(ps.isfiring,false)
  fill!(ps.isrefractory,false)
  return nothing
end

# Connection : ConnGeneralIF, defined in firingneruons_shared.jl

function local_update!(t_now::Float64,dt::Float64,ps::PSLIF)
	# computes the update to internal voltage, given the total input
  # dv =  (v_leak - v ) dt / τ + I dt / Cap
  copy!(ps.alloc_dv,ps.state_now)  # v
  ps.alloc_dv .-= ps.input  # (v-I)
  lmul!(-dt/ps.neurontype.τ,ps.alloc_dv) # du =  dt/τ (-v+I)
  ps.state_now .+= ps.alloc_dv # v_{t+1} = v_t + dv
	# update spikes and refractoriness, and end
  # function defined in firingneruons_shared.jl
  return _spiking_state_update!(ps.state_now,ps.isfiring,ps.isrefractory,ps.last_fired,
    t_now,ps.neurontype.τ_refractory,ps.neurontype.v_threshold,ps.neurontype.v_reset)
end


# analytic stuff

function expected_period_norefr(neu::NTLIF,input)
	return expected_period_norefr(neu.τ,neu.v_reset,neu.v_threshold,input)
end

function expected_period_norefr(τ::Real,vr::Real,vth::Real,input::Real)
	return τ*log( (input-vr)/(input-vth) )
end


# forward_signal(...) defined in firingneruons_shared.jl


#=
# threshold-linear input-output function
struct PopLIF <: Population
  τ::Float64 # time constant
	v_threshold::Float64 # fixed spiking threshold
	v_reset::Float64 # reset after spike
	τ_refractory::Float64 # refractory time
	τ_output_decay::Float64 # decay of postsynaptic currents
end

struct PSLIF{P} <: PopulationState
  population::P
  state_now::Vector{Float64}
  input::Vector{Float64}
	alloc_dv::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
end
function PSLIF(p::Population)
  PSLIF(p, ntuple(_-> zeros(Float64,p.n),4)...,ntuple(_-> falses(p.n),2)...)
end


struct ConnectionLIF <: Connection
  postps::PSLIF # postsynaptic population state
  preps::PSLIF # presynaptic population state
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
  post_current::Vector{Float64}
end

function ConnectionLIF(post::PSLIF,weights::SparseMatrixCSC,pre::PSLIF)
  aR,aC,_ = findnz(weights)
	npost = post.population.n
  ConnectionLIF(post,pre,aR,aC,weights,zeros(Float64,npost))
end


function dynamics_step!(t_now::Float64,dt::Float64,ps::PSLIF)
	# computes the update to internal voltage, given the total input
  copy!(ps.alloc_dv,ps.state_now)  # v
  ps.alloc_dv .-= ps.input  # (v-I)
  lmul!(-dt/ps.population.τ,ps.alloc_dv) # du =  dt/τ (-v+I)
  ps.state_now .+= ps.alloc_dv # v_{t+1} = v_t + dv
	# now spikes and refractoriness
	reset_spikes!(ps)
	@inbounds @simd for i in eachindex(ps.state_now)
		if ps.state_now[i] > ps.population.v_threshold
			ps.isfiring[i] = true
			ps.state_now[i] =  ps.population.v_reset
			ps.last_fired[i] = t_now
			ps.isrefractory[i] = true
		# check only when refractory
		elseif ps.isrefractory[i] && 
				( (t_now-ps.last_fired[i]) >= ps.population.τ_refractory)
			ps.isrefractory[i] = false
		end
	end
	# 
  return nothing
end

function send_signal!(t_now::Float64,conn::ConnectionLIF)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	τ_decay = conn.preps.population.τ_output_decay
	for _pre in findall(conn.preps.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			post_idx = post_idxs[_pnz]
			# update the post currents even when refractory
				conn.post_current[post_idx] += 
						weightsnz[_pnz] / τ_decay
		end
	end
	# finally , add the currents to postsynaptic input
	# ONLY non-refractory neurons
	post_refr = findall(conn.postps.isrefractory) # refractory ones
	@inbounds @simd for i in eachindex(conn.postps.input)
		if !(i in post_refr)
			conn.postps.input[i] += conn.post_current[i]
		end
	end
  return nothing
end

# postsynaptic currents decay in time
function dynamics_step!(t_now::Real,dt::Real,conn::ConnectionLIF)
	τ_decay = conn.preps.population.τ_output_decay
	@inbounds @simd for i in eachindex(conn.post_current)
		conn.post_current[i] -= dt*conn.post_current[i] / τ_decay
	end
	return nothing
end

@inline function send_signal!(t::Real,input::PopInputStatic{P}) where P<:PSLIF
	refr = findall(input.population_state.isrefractory) # refractory ones
  @inbounds @simd for i in eachindex(input.population_state.input)
		if !(i in refr)
  		input.population_state.input[i] += input.h[i]
		end
	end
  return nothing
end



		for cc = 1:Ncells
    ###plasticity detectors, forward euler###
		  x[cc]  += -dt*x[cc] /p.tau_i #inhibitory traces
			o1[cc] += -dt*o1[cc]/p.tau_minus #triplet terms
			o2[cc] += -dt*o2[cc]/p.tau_y
			r1[cc] += -dt*r1[cc]/p.tau_plus
			r2[cc] += -dt*r2[cc]/p.tau_x

      ## NEURON DYNAMICS
			if t > (lastSpike[cc] + p.taurefrac) #only after refractory period
        ###CONDUCTANCE DYNAMICS####
			  #forward euler
				Ie[cc] += -dt*Ie[cc]/p.tauedecay
				Ip[cc] += -dt*Ip[cc]/p.taupdecay
				Is[cc] += -dt*Is[cc]/p.tausdecay
				Iv[cc] += -dt*Iv[cc]/p.tauvdecay
        ###CONDUCTANCE DYNAMICS END###

        ###LIF NEURONS###
				#forward euler
				if cc <= p.Ne #excitatory neuron
					dv =  -v[cc]/p.taue + Ie[cc] - Ip[cc] - Is[cc] +  I_stim[cc]+ I_E0[cc]
					v[cc] += dt*dv
					if  v[cc] > p.vth_e
						spiked[cc] = true
					end
				elseif p.Ne < cc <= (p.Ne+p.Np) #pv neurons
					dv = -v[cc]/p.taup + Ie[cc] - Ip[cc] - Is[cc] + I_stim[cc] + I_P0[cc]
					v[cc] += dt*dv
					if v[cc] > p.vth_i
						spiked[cc] = true
					end
				elseif (p.Ne+p.Np) < cc <= (p.Ne+p.Np+p.Ns)#sst neurons
					dv = -v[cc]/p.taus + Ie[cc] -Iv[cc]+ I_S0[cc] + I_stim[cc]
					v[cc] += p.dt*dv
					if v[cc] > p.vth_i
						spiked[cc] = true
					end
				else #vip neurons
					dv = -v[cc]/p.tauv + Ie[cc] -Is[cc] -Ip[cc] + I_V0[cc] + I_stim[cc]
					v[cc] += dt*dv
					if v[cc] > p.vth_i
						spiked[cc] = true
					end
				end
        ###LIF NEURONS END###

        ###UPDATE WHEN SPIKE OCCURS
				if spiked[cc] #spike occurred
					v[cc] = p.vre #voltage back to reset potential
					lastSpike[cc] = t #record last spike time
					ns[cc] += 1 #lists number of spikes per neuron
					if ns[cc] <= p.Nspikes #spike time are only record for ns < Nspikes
						times[cc,ns[cc]] = t #recording spiking times
					end
					# for synaptic updates other than input upon every presynaptic spike
					for dd = 1:Ncells
						if cc <= p.Ne
							Ie[dd]+= weights[cc,dd]/p.tauedecay
						elseif p.Ne < cc <= (p.Np+p.Ne)
							Ip[dd]+= weights[cc,dd]/p.taupdecay
						elseif (p.Ne+p.Np) < cc <=(p.Np+p.Ne+p.Ns)
							Is[dd]+= weights[cc,dd]/p.tausdecay
						else
							Iv[dd]+= weights[cc,dd]/p.tauvdecay
						end
					end
				end #end if(spiked)
			end #end if(not refractory)

=#