
abstract type PoissonIO end
struct PoissonIOReLU <: PoissonIO end
struct PoissonIOMaxed <: PoissonIO 
  maxrate::Float64
end

@inline function (pio::PoissonIOReLU)(rnow::Float64)
  if rnow > 0.0
    return rnow
  else
    return 0.0
  end
end
@inline function (pio::PoissonIOMaxed)(rnow::Float64)
  if rnow < 0.0
    return 0.0
  elseif rnow > pio.maxrate
    return maxrate
  else
    return rnow
  end
end

@inline function apply_io!(rout::Vector{R},rnow::Vector{R},::PoissonIOReLU) where R
  @inbounds @simd for i in eachindex(rout)
    rn = rnow[i]
    rout[i] = rn < zero(R) ? zero(R) : rn
  end
  return nothing
end



struct PSPoissonNeuron{PIO<:PoissonIO} <: PSSpiking
  τ::Float64
  n::Int64
  io_function::PIO
  state_now::Vector{Float64}
  input::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
  random_alloc::Vector{Float64}
  rate_now::Vector{Float64}
  function PSPoissonNeuron(τ::Float64,n::Integer;(io_function::PIO)=PoissonIOReLU()) where PIO
    input = fill(0.0,n)
    state_now = fill(0.0,n)
    last_fired = fill(-Inf,n)
    isfiring = fill(false,n)
    random_alloc = fill(NaN,n)
    rate_now = fill(NaN,n)
    return new{PIO}(τ,n,io_function,input,state_now,last_fired,isfiring,random_alloc,rate_now)
  end
end
function reset!(ps::PSPoissonNeuron)
  fill!(ps.input,0.0)
  fill!(ps.state_now,0.0)
  fill!(ps.last_fired,-Inf)
  fill!(ps.isfiring,false)
  return nothing
end

@inline function reset_input!(ps::PSPoissonNeuron)
  fill!(ps.input,0.0)
  return nothing
end

function local_update!(::Float64,dt::Float64,ps::PSPoissonNeuron)
  # one step forward in time
  @. ps.state_now += (dt/ps.τ) *(-ps.state_now+ps.input)
  # compute firing probability
  apply_io!(ps.rate_now,ps.state_now,ps.io_function)
  ps.rate_now .*= dt
  # refresh randomly generated values
  rand!(ps.random_alloc)
  @. ps.isfiring = ps.random_alloc < ps.rate_now 
  # all done !
  return nothing 
end

# function local_update_old!(::Float64,dt::Float64,ps::PSPoissonNeuron)
#   # refresh randomly generated values
#   rand!(ps.random_alloc)
#   @inbounds @simd for i in 1:ps.n
#     # euler step here...
#     # rate is total input filtered by nonlinearity
#     state_now = ps.state_now[i]
#     state_now += dt * (-state_now+ps.input[i]) / ps.τ
#     rate_now = ps.io_function(state_now)
#     # is firing ?
#     ps.isfiring[i] = ps.random_alloc[i] < ( rate_now * dt)
#     # store instantaneopus rate
#     ps.state_now[i] = state_now
#   end
#   #all done !
#   return nothing 
# end



abstract type PoissonConnectionSign end
struct PoissonExcitatory <: PoissonConnectionSign end
struct PoissonInhibitory <: PoissonConnectionSign end

# The kernel is a connection property
# input traces are stored in there, too!
abstract type AbstractConnectionPoisson <: AbstractBaseConnection end

struct ConnectionPoissonExpKernel{N,PL<:NTuple{N,PlasticityRule},S<:PoissonConnectionSign} <: AbstractConnectionPoisson
  sign::S
  post_trace::Trace
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::PL
end
function ConnectionPoissonExpKernel(sign::PoissonConnectionSign,τ::Float64,
    weights::Union{Matrix{Float64},SparseMatrixCSC{Float64,Int64}};
    plasticities=(NoPlasticity(),))
  if weights isa Matrix 
    weights = sparse(weights)
  end
  npost = size(weights,1)
  post_trace = Trace(τ,npost)
  return ConnectionPoissonExpKernel(sign,post_trace,weights,plasticities)
end

@inline function kernel_decay!(co::ConnectionPoissonExpKernel,dt::Float64)
  trace_decay!(co.post_trace,dt)
  return nothing
end
@inline function poisson_kernel_trace_update!(co::ConnectionPoissonExpKernel,wij::Float64,
    idx_post::Integer)
  co.post_trace[idx_post] += wij/co.post_trace.τ  
  return nothing
end

# difference between E and I goes here!
@inline function add_signal_to_input!(input::Vector{Float64},
    conn::ConnectionPoissonExpKernel{A,B,PoissonExcitatory}) where {A,B}
  input .+= conn.post_trace.val
  return nothing
end
@inline function add_signal_to_input!(input::Vector{Float64},
    conn::ConnectionPoissonExpKernel{A,B,PoissonInhibitory}) where {A,B}
  input .-= conn.post_trace.val
  return nothing
end


# Forward signals that arrive in the form of spikes 
function forward_signal!(::Real,dt::Real,
      pspost::PSPoissonNeuron,conn::AbstractConnectionPoisson,pspre::PSSpiking)
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
  # add signal to post trace
	for _pre in findall(pspre.isfiring)
		_posts_nz = nzrange(conn.weights,_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
			idx_post = post_idxs[_pnz]
      poisson_kernel_trace_update!(conn,weightsnz[_pnz],idx_post)
		end
	end
  # add ALL traces to input
  add_signal_to_input!(pspost.input,conn)
  # finally, all postsynaptic conductances decay in time
  kernel_decay!(conn,dt)
  # all done!
  return nothing
end


# deals with inputs that are just currents
struct PoissonInputCurrentConstant{V<:Union{Float64,Vector{Float64}}} <: PopulationState
  current::V
end
struct PoissonInputCurrentFunVector <: PopulationState
  f::Function # f(::Float64) -> Array{Float64}
end
struct PoissonInputCurrentNormal <: PopulationState
  μ::Float64
  σ::Float64
  rand_alloc::Vector{Float64}
end

@inline function forward_signal!(::Real,::Real,
        pspost::PSPoissonNeuron,conn::FakeConnection,
        pspre::PoissonInputCurrentConstant)
  pspost.input .+= pspre.current
  return nothing
end

@inline function forward_signal!(t_now::Real,::Real,
        pspost::PSPoissonNeuron,conn::FakeConnection,
        pspre::PoissonInputCurrentFunVector)
  current::Vector{Float64} = pspre.f(t_now)
  pspost.input .+= current
  return nothing
end
function forward_signal!(::Real,dt::Real,
      pspost::PSPoissonNeuron,conn::FakeConnection,
      pspre::PoissonInputCurrentNormal)
  # generate random values    
  randn!(pspre.rand_alloc)
  # regularize so that std is σ for isolated neuron
  σstar = sqrt(2*pspost.τ/dt)*pspre.σ
  @. pspost.input += pspre.μ+σstar*pspre.rand_alloc
  return nothing
end