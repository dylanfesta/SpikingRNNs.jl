# ./src/lif_conductance.jl

############
# conductance-based (classic) LIF

# threshold-linear input-output function
struct PopLIFCO <: Population
  n::Int64 # pop size
  τ::Float64 # time constant
	v_threshold::Vector{Float64} # spiking threshold can vary
	v_reset::Float64 # reset after spike
	τ_refractory::Float64 # refractory time
	τ_post_conductance_decay::Float64 # decay of postsynaptic conductance
	post_reversal_potential::Float64 # reversal potential that affects postsynaptic neurons
end

struct PSLIFCO{P} <: PopulationState
  population::P
  state_now::Vector{Float64}
  input::Vector{Float64}
	alloc_dv::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
	pre_reverse_potentials::Vector{Float64}
	pre_conductances_now::Vector{Float64}
end
function PSLIFCO(p::Population)
  PSLIF(p, ntuple(_-> zeros(Float64,p.n),4)...,ntuple(_-> falses(p.n),2)...)
end


struct ConnectionLIFCO <: Connection
  postps::PSLIFCO # postsynaptic population state
  preps::PSLIFCO # presynaptic population state
  adjR::Vector{Int64} # adjacency matrix, rows
  adjC::Vector{Int64} # adjacency matrix, columns
  weights::SparseMatrixCSC{Float64,Int64}
  post_current::Vector{Float64}
end

function ConnectionLIFCO(post::PSLIFCO,weights::SparseMatrixCSC,pre::PSLIFCO)
  aR,aC,_ = findnz(weights)
	npost = post.population.n
  ConnectionLIFCO(post,pre,aR,aC,weights,zeros(Float64,npost))
end


function dynamics_step!(t_now::Float64,dt::Float64,ps::PSLIFCO)
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

function send_signal!(t_now::Float64,conn::ConnectionLIFCO)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
	τ_decay = conn.preps.population.τ_post_current_decay
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
function dynamics_step!(t_now::Real,dt::Real,conn::ConnectionLIFCO)
	τ_decay = conn.preps.population.τ_post_current_decay
	@inbounds @simd for i in eachindex(conn.post_current)
		conn.post_current[i] -= dt*conn.post_current[i] / τ_decay
	end
	return nothing
end

@inline function send_signal!(t::Real,input::PopInputStatic{P}) where P<:PSLIFCO
	refr = findall(input.population_state.isrefractory) # refractory ones
  @inbounds @simd for i in eachindex(input.population_state.input)
		if !(i in refr)
  		input.population_state.input[i] += input.h[i]
		end
	end
  return nothing
end


#=

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