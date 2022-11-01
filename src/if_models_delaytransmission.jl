
abstract type AbstractConnectionIFDelayed <: AbstractConnectionIF end

function find_pre_receiving(conn::AbstractConnectionIFDelayed,pre_isfiring::BitArray{1})
  # if firing, start the delay countdown
  conn.delay_counter[pre_isfiring] .= conn.transmission_delay
  # then find those with countdown zero
  return findall(iszero,conn.delay_counter)
end


@inline function forward_signal_end!(conn::AbstractConnectionIFDelayed,dt::Real)
  kernel_decay!(conn.synaptic_kernel,dt)
  conn.delay_counter .-= 1
  return nothing
end