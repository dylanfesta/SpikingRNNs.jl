
abstract type PlasticityBounds end

struct PlasticityBoundsNone <: PlasticityBounds  end

struct PlasticityBoundsNonnegative <: PlasticityBounds end

struct PlasticityBoundsLowHigh <: PlasticityBounds 
  low::Float64
  high::Float64
end


function (plb::PlasticityBoundsNonnegative)(w::R,Δw::R) where R
  ret = w+Δw
  if ret < zero(R)
    return  zero(R)
  else 
    return ret
  end
end
function (plb::PlasticityBoundsLowHigh)(w::R,Δw::R) where R
  ret = w+Δw
  return min(plb.high,max(plb.low,ret))
end

# no plasticity, nothing to do
@inline function plasticity_update!(::Real,::Real,
     ::PopulationState,::Connection,::PopulationState,::NoPlasticity)
  return nothing
end

@inline function plasticity_update!(::Real,::Real,
     ::PSSpikingType,::Connection,::PSSpikingType,::NoPlasticity)
  return nothing
end

# Pairwise plasticity 
"""
    function PairSTDP(τplus,τminus,Aplus,Aminus,n_post,n_pre;
         plasticity_bounds=PlasticityBoundsNonnegative())

constructor for "classic" pairwise STPD rule.
The `Aminus` coefficient, when positive, causes a *reduction*
in the weights. 
"""
struct PairSTDP <: PlasticityRule
  Aplus::Float64
  Aminus::Float64
  tracerplus::Trace 
  traceominus::Trace
  bounds::PlasticityBounds
  function PairSTDP(τplus,τminus,Aplus,Aminus,n_post,n_pre;
       plasticity_bounds=PlasticityBoundsNonnegative())
    new(Aplus,Aminus,
      Trace(τplus,n_pre),
      Trace(τminus,n_post),
      plasticity_bounds)
  end
end

function reset!(pl::PairSTDP)
  reset!(pl.traceplus)
  reset!(pl.traceminus)
  return nothing
end

function plasticity_update!(::Real,dt::Real,
     pspost::PSSpikingType,conn::Connection,pspre::PSSpikingType,
     plast::PairSTDP)
  # elements of sparse matrix that I need
  _colptr = SparseArrays.getcolptr(conn.weights) # column indexing
	row_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
  idx_pre_spike = findall(pspre.isfiring) 
  idx_post_spike = findall(pspost.isfiring) 
  # update synapses
  # presynpatic spike go along w column
  for j_pre in idx_pre_spike
		_posts_nz = nzrange(conn.weights,j_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
      ipost = row_idxs[_pnz]
      Δw = -plast.traceominus[ipost]*plast.Aminus
      weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
    end
  end
  # postsynaptic spike: go along w row
  # innefficient ... need to search i element for each column
  for i_post in idx_post_spike
    for j_pre in (1:pspre.n)
      _start = _colptr[j_pre]
      _end = _colptr[j_pre+1]-1
      _pnz = searchsortedfirst(row_idxs,i_post,_start,_end,Base.Order.Forward)
      if (_pnz<=_end) && (row_idxs[_pnz] == i_post) # must check!
        Δw = plast.tracerplus[j_pre]*plast.Aplus 
        weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
      end
    end
  end
  # update the plasticity trace variables
  for j_pre in idx_pre_spike
    plast.tracerplus[j_pre]+=1.0
  end
  for i_post in idx_post_spike
    plast.traceominus[i_post]+=1.0
  end
  # time decay
  trace_decay!(plast.tracerplus,dt)
  trace_decay!(plast.traceominus,dt)
  return nothing
end



### Triplets rule
"""
  PlasticityTriplets(τplus,τminus,τx,τy,
      A2plus,A3plus,A2minus,A3minus,n_post,n_pre;
      plasticity_bounds=PlasticityBoundsNonnegative())

it construc!      
"""
struct PlasticityTriplets <: PlasticityRule
  A2plus::Float64
  A3plus::Float64
  A2minus::Float64
  A3minus::Float64
  trace2rplus::Trace # a.k.a r1
  trace3oplus::Trace # a.k.a o2
  trace2ominus::Trace # a.k.a o1
  trace3rminus::Trace # a.k.a r2
  bounds::PlasticityBounds
  function PlasticityTriplets(τplus,τminus,τx,τy,
      A2plus,A3plus,A2minus,A3minus,n_post,n_pre;
        plasticity_bounds=PlasticityBoundsNonnegative())
    trace_pre(τ) = Trace(τ,n_pre)
    trace_post(τ) = Trace(τ,n_post)
    new(A2plus,A3plus,A2minus,A3minus,
      Trace(τplus,n_pre),
      Trace(τy,n_post),
      Trace(τminus,n_post),
      Trace(τx,n_pre),
      plasticity_bounds)
  end
end

function reset!(pl::PlasticityTriplets)
  reset!.((pl.trace2rplus,pl.trace2ominus,pl.trace3oplus,pl.trace3rminus))
  return nothing
end

function plasticity_update!(::Real,dt::Real,
     pspost::PSSpikingType,conn::Connection,pspre::PSSpikingType,
     plast::PlasticityTriplets)
  # elements of sparse matrix that I need
  _colptr = SparseArrays.getcolptr(conn.weights) # column indexing
	row_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
  idx_pre_spike = findall(pspre.isfiring) 
  idx_post_spike = findall(pspost.isfiring) 
  # update the pairwise plasticity traces 
  for j_pre in idx_pre_spike
    plast.trace2rplus[j_pre]+=1.0 
  end
  for i_post in idx_post_spike
    plast.trace2ominus[i_post]+=1.0
  end
  # update synapses
  # presynpatic spike go along w column
  for j_pre in idx_pre_spike
		_posts_nz = nzrange(conn.weights,j_pre) # indexes of corresponding pre in nz space
    trace3_jpre = plast.trace3rminus[j_pre]
		@inbounds for _pnz in _posts_nz
      i_post = row_idxs[_pnz]
      Δw = -plast.trace2ominus[i_post]*(plast.A2minus+plast.A3minus*trace3_jpre)
      weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
    end
  end
  # postsynaptic spike: go along w row
  # innefficient ... need to search i element for each column
  for i_post in idx_post_spike
    trace3_ipost = plast.trace3oplus[i_post]
    for j_pre in (1:pspre.n)
      _start = _colptr[j_pre]
      _end = _colptr[j_pre+1]-1
      _pnz = searchsortedfirst(row_idxs,i_post,_start,_end,Base.Order.Forward)
      if (_pnz<=_end) && (row_idxs[_pnz] == i_post) # must check!
        Δw = plast.trace2rplus[j_pre]*(plast.A2plus+plast.A3plus*trace3_ipost)
        weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
      end
    end
  end
  # update the plasticity traces 2
  for j_pre in idx_pre_spike
    plast.trace3rminus[j_pre]+=1.0
  end
  for i_post in idx_post_spike
    plast.trace3oplus[i_post]+=1.0
  end
  # and time decay
  trace_decay!.( (plast.trace2rplus,plast.trace2ominus,
                  plast.trace3rminus,plast.trace3oplus) ,dt)
  return nothing
end

# Vogels et al 2011 plasticity rule

struct PlasticityInhibitoryVogels <: PlasticityRule
  η::Float64
  α::Float64
  traceo::Trace
  tracer::Trace
  bounds::PlasticityBounds
  function PlasticityInhibitoryVogels(τ,η,n_post,n_pre;
      r_target=5.0,
      plasticity_bounds::PlasticityBounds=PlasticityBoundsNonnegative())
    α = 2*r_target*τ
    new(η,α,
      Trace(τ,n_post),
      Trace(τ,n_pre),
      plasticity_bounds)
  end
end
function reset!(pl::PlasticityInhibitoryVogels)
  reset!(pl.traceo)
  reset!(pl.tracer)
  return nothing
end

function plasticity_update!(::Real,dt::Real,
     pspost::PSSpikingType,conn::Connection,pspre::PSSpikingType,
     plast::PlasticityInhibitoryVogels)
  # elements of sparse matrix that I need
  _colptr = SparseArrays.getcolptr(conn.weights) # column indexing
	row_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
  idx_pre_spike = findall(pspre.isfiring) 
  idx_post_spike = findall(pspost.isfiring) 
  # update traces first  (mmmh)
  for j_pre in idx_pre_spike
    plast.tracer[j_pre]+=1.0
  end
  for i_post in idx_post_spike
    plast.traceo[i_post]+=1.0
  end
  # update synapses
  # presynpatic spike go along w column
  for j_pre in idx_pre_spike
		_posts_nz = nzrange(conn.weights,j_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
      ipost = row_idxs[_pnz]
      Δw = plast.η*(plast.traceo[ipost]-plast.α)
      weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
    end
  end
  # postsynaptic spike: go along w row
    # innefficient ... need to search i element for each column
  for i_post in idx_post_spike
    for j_pre in (1:pspre.n)
      _start = _colptr[j_pre]
      _end = _colptr[j_pre+1]-1
      _pnz = searchsortedfirst(row_idxs,i_post,_start,_end,Base.Order.Forward)
      if (_pnz<=_end) && (row_idxs[_pnz] == i_post) # must check!
        Δw = plast.η*plast.tracer[j_pre]
        weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
      end
    end
  end
  trace_decay!(plast.traceo,dt)
  trace_decay!(plast.tracer,dt)
  return nothing
end
