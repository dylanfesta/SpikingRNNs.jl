# ./src/lif_conductance.jl

############
# conductance-based (classic) LIF

# threshold-linear input-output function
# WARNING !
# TODO  !  τ_output_decay is missing!
struct NTEIF <: NeuronType
  τ::Float64 # time constant (membrane capacitance)
  g_l::Float64 # conductance leak
  v_expt::Float64 # exp term threshold
  steep_exp::Float64 # steepness of exponential term
  v_threshold::Float64 # spiking threshold 
  v_reset::Float64 # reset after spike
  v_leak::Float64 # reversal potential for leak term
  τ_refractory::Float64 # refractory time
end

struct PSEIF{NT} <: PSSpikingType{NT}
  neurontype::NT
  n::Int64 # pop size
  state_now::Vector{Float64}
  input::Vector{Float64}
	alloc_dv::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
	isrefractory::BitArray{1}
  function PSEIF(p::NTEIF,n::Integer)
    zz = _ -> zeros(Float64,n)
    ff = _ -> falses(n)
    new{NTEIF}(p,n,ntuple(zz,4)...,ntuple(ff,2)...)
  end
end



# Connection : ConnGeneralIF, defined in firingneruons_shared.jl

function local_update!(t_now::Float64,dt::Float64,ps::PSEIF)
	# computes the update to internal voltage, given the total input
  # dv = (dt / τ) * (g_l (v_leak - v ) + g_l * steep_exp * exp((v-v_expt)/steep_exp)
  #  + input ) 
  dttau =  dt / ps.neurontype.τ
  @. ps.alloc_dv =  dttau * ( ps.neurontype.g_l*(ps.neurontype.v_leak-ps.state_now)
    + ps.neurontype.steep_exp*ps.neurontype.g_l*exp(ps.state_now-ps.neurontype.v_expt)
    + ps.input)
  ps.state_now .+= ps.alloc_dv # v_{t+1} = v_t + dv
	# update spikes and refractoriness, defined in firingneruons_shared.jl
  return _spiking_state_update!(ps.state_now,ps.isfiring,ps.isrefractory,ps.last_fired,
    t_now,ps.neurontype.τ_refractory,ps.neurontype.v_threshold,ps.neurontype.v_reset)
end


# forward_signal(...) defined in firingneruons_shared.jl


