
# examines state_now, if above threshold, isfiring and refractory turn true
# and potential is reset. Finally, removes expired refractoriness
function _spiking_state_update!(state_now::Vector{R},
    isfiring::BitArray{1},isrefractory::BitArray{1},
    last_fired::Vector{R},
    t_now::R, t_refractory::R,
    v_threshold::R, v_reset::R) where R<:Real
  reset_spikes!(isfiring)  
	@inbounds @simd for i in eachindex(state_now)
		if state_now[i] > v_threshold
			state_now[i] =  v_reset
			isfiring[i] = true
			last_fired[i] = t_now
			isrefractory[i] = true
		# check only when refractory
		elseif isrefractory[i] && 
				( (t_now-last_fired[i]) >= t_refractory)
			isrefractory[i] = false
		end
	end
  return nothing
end