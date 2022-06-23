abstract type PoissonRole end
struct PoissonExcitatory <: PoissonNeuronRole end
struct PoissonInhibitory <: PoissonNeuronRole end

abstract type PoissonIO end
struct PoissonIOReLU <: PoissonIO end

abstract type PoissonGeneralInput end

struct PoissonFixedInput <: PoissonGeneralInput
  input::Vector{Float64}
end

@inline function (pio::PoissonIOReLU)(rnow::Float64)
  if rnow > 0.0
    return rnow
  else
    return 0.0
  end
end

struct PSPoisson{PR<:PoissonRole,PIO<:PoissonIO} <: PSSpiking
  role::PR
  io_function::PIO
  n::Int64
  traces::NTuple{N,Trace} where N
  input::Vector{Float64}
  state_now::Vector{Float64}
	last_fired::Vector{Float64}
	isfiring::BitArray{1}
  random_alloc::Vector{Float64}
  function PSPoisson(n::Integer,role::PR,traces;(io_function::PIO)=PoissonIOReLU()) where {PR,PIO}
    input = fill(0.0,n)
    state_now = fill(0.0,n)
    last_fired = fill(-Inf,n)
    isfiring = fill(false,n)
    random_alloc = fill(NaN,n)
    return new{PR,PIO}(role,io_function,n,traces,input,state_now,last_fired,isfiring,random_alloc)
  end
end
function reset!(ps::PSPoisson)
  fill!(ps.input,0.0)
  fill!(ps.state_now,0.0)
  fill!(ps.last_fired,-Inf)
  fill!(ps.isfiring,false)
  reset!.(ps.traces)
  return nothing
end

@inline function reset_input!(ps::PSPoisson)
  fill!(ps.input,0.0)
  fill!(ps.state_now,0.0)
  return nothing
end

# tau has one element for exp kernel, but two elements for sum of exp shape...
function population_state_and_traces(n::Integer,role::PoissonRole,τs::NTuple{N,Float64};
    (io_function::PoissonIO)=PoissonIOReLU()) where N
  traces = Trace.(τs,n)
  return PSPoisson(n,role,traces;io_function=io_function),traces
end


function local_update!(::Float64,dt::Float64,ps::PSPoisson)
  # refresh randomly generated values
  rand!(ps.random_alloc)
  @inbounds @simd for i in 1:ps.n
    # rate is total input filtered by nonlinearity
    state_now = ps.io_function(ps.input[i])
    # is firing ?
    if ps.random_alloc[i] > (state_now * dt)
    # then update trace!
      for tra in ps.traces
        tra[i] += 1.0
      end
      ps.isfiring[i] = true
    else
      ps.isfiring[i] = false
    end
    ps.state_now[i] = state_now # this is only for data storage and visualizaton purpose
  end
  # move traces one step forward
  trace_decay!.(ps.traces,dt)
  # all done !
  return nothing 
end


# The kernel is a connection property... but traces must be stored stored in the population state!
# (so that they can be updated at each dt)

struct ConnectionPoissonExpKernel{N,PL<:NTuple{N,PlasticityRule}} <: BaseConnection
  alloc_input::Vector{Float64}
  weights::SparseMatrixCSC{Float64,Int64}
  plasticities::PL
end

# these are actually just matrix-vector products!
function forward_signal!(::Float64,::Float64,pspost::PSPoisson,
    conn::ConnectionPoissonExpKernel,pspre::PSPoisson{PoissonExcitatory,T where T})
  mul!(conn.alloc_input,conn.weights,pspre.traces[1].val)
  pspost.input .+= conn.alloc_input
  return nothing
end
function forward_signal!(::Float64,::Float64,pspost::PSPoisson,
    conn::ConnectionPoissonExpKernel,pspre::PSPoisson{PoissonInhibitory,T where T})
  mul!(conn.alloc_input,conn.weights,pspre.traces[1].val)
  pspost.input .-= conn.alloc_input
  return nothing
end

# useful for more general cases ?
function forward_signal!(::Float64,::Float64,pspost::PSPoisson,
    conn::BaseConnection,pspre::PSPoissonGeneralInput)
  mul!(conn.alloc_input,conn.weights,pspre.trace)
  pspost.input .+= conn.alloc_input
  return nothing
end
function forward_signal!(::Float64,::Float64,pspost::PSPoisson,
    conn::FakeConnection,pspre::PSPoissonGeneralInput)
  pspost.input .+= pspre.input
  return nothing
end
