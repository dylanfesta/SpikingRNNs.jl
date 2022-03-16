
abstract type PlasticityBounds end

struct PlasticityBoundsNone <: PlasticityBounds  end

struct PlasticityBoundsNonnegative <: PlasticityBounds end

struct PlasticityBoundsLowHigh <: PlasticityBounds 
  low::Float64
  high::Float64
end


function (plb::PlasticityBoundsNonnegative)(w::R,Δw::R) where R
  ret = w+Δw
  return min(0.0,ret)
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
struct PairSTDP <: PlasticityRule
  τplus::Float64
  τminus::Float64
  Aplus::Float64
  Aminus::Float64
  o::Vector{Float64} # pOst
  r::Vector{Float64} # pRe
  bounds::PlasticityBounds
  function PairSTDP(τplus,τminus,Aplus,Aminus,n_post,n_pre;
       plasticity_bounds=PlasticityBoundsNonnegative())
    new(τplus,τminus,Aplus,Aminus,zeros(n_post),zeros(n_pre),plasticity_bounds)
  end
end

function reset!(pl::PairSTDP)
  fill!(pl.r,0.0)
  fill!(pl.o,0.0)
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
      Δw = -plast.o[ipost]*plast.Aminus
      weightsnz[_pnz] = max(0.0,weightsnz[_pnz]-Δw)
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
        Δw = plast.r[j_pre]*plast.Aplus 
        weightsnz[_pnz]+= Δw
      end
    end
  end
  # update the plasticity trace variables
  for j_pre in idx_pre_spike
    plast.r[j_pre]+=1.0
  end
  for i_post in idx_post_spike
    plast.o[i_post]+=1.0
  end
  @. plast.r -= plast.r*dt/plast.τplus
  @. plast.o -= plast.o*dt/plast.τminus
  return nothing
end


### Triplets rule
struct PlasticityTriplets <: PlasticityRule
  τplus::Float64
  τminus::Float64
  τx::Float64
  τy::Float64
  A2plus::Float64
  A3plus::Float64
  A2minus::Float64
  A3minus::Float64
  o1::Vector{Float64} # pOst
  o2::Vector{Float64} # pOst
  r1::Vector{Float64} # pRe
  r2::Vector{Float64} # pRe
  bounds::PlasticityBounds
  function PlasticityTriplets(τplus,τminus,τx,τy,
      A2plus,A3plus,A2minus,A3minus,n_post,n_pre;
        plasticity_bounds=PlasticityBoundsNonnegative())
    new(τplus,τminus,τx,τy,A2plus,A3plus,A2minus,A3minus,
        ntuple(_->zeros(n_post),2)...,
        ntuple(_->zeros(n_pre),2)...,plasticity_bounds)
  end
end

function reset!(pl::PlasticityTriplets)
  fill!(pl.r1,0.0)
  fill!(pl.r2,0.0)
  fill!(pl.o1,0.0)
  fill!(pl.o2,0.0)
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
  # update the plasticity traces 1
  for j_pre in idx_pre_spike
    plast.r1[j_pre]+=1.0 
  end
  for i_post in idx_post_spike
    plast.o1[i_post]+=1.0
  end
  # update synapses
  # presynpatic spike go along w column
  for j_pre in idx_pre_spike
		_posts_nz = nzrange(conn.weights,j_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
      i_post = row_idxs[_pnz]
      Δw = -plast.o1[i_post]*(plast.A2minus+plast.A3minus*plast.r2[j_pre])
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
        Δw = plast.r1[j_pre]*(plast.A2plus+plast.A3plus*plast.o2[i_post])
        weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
      end
    end
  end
  # update the plasticity traces 2
  for j_pre in idx_pre_spike
    plast.r2[j_pre]+=1.0
  end
  for i_post in idx_post_spike
    plast.o2[i_post]+=1.0
  end
  # and timestep all
  @. plast.r1 -= plast.r1*dt/plast.τplus
  @. plast.r2 -= plast.r2*dt/plast.τx
  @. plast.o1 -= plast.o1*dt/plast.τminus
  @. plast.o2 -= plast.o2*dt/plast.τy
  return nothing
end


# Vogels et al 2011 plasticity rule

struct PlasticityInhibitoryVogels <: PlasticityRule
  τ::Float64
  η::Float64
  α::Float64
  o::Vector{Float64} # pOst
  r::Vector{Float64} # pRe
  bounds::PlasticityBounds
  function PlasticityInhibitoryVogels(τ,η,n_post,n_pre;
      r_target=5.0,
      plasticity_bounds::PlasticityBounds=PlasticityBoundsNonnegative())
    α = 2*r_target*τ
    new(τ,η,α,zeros(n_post),zeros(n_pre),plasticity_bounds)
  end
end
function reset!(pl::PlasticityInhibitoryVogels)
  fill!(pl.r,0.0)
  fill!(pl.o,0.0)
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
    plast.r[j_pre]+=1.0
  end
  for i_post in idx_post_spike
    plast.o[i_post]+=1.0
  end
  # update synapses
  # presynpatic spike go along w column
  for j_pre in idx_pre_spike
		_posts_nz = nzrange(conn.weights,j_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
      ipost = row_idxs[_pnz]
      Δw = plast.η*(plast.o[ipost]-plast.α)
      #@show plast.o[ipost]
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
        Δw = plast.η*plast.r[j_pre]
        #@show Δw
        weightsnz[_pnz] = plast.bounds(weightsnz[_pnz],Δw)
      end
    end
  end
  @. plast.r -= plast.r*dt/plast.τ
  @. plast.o -= plast.o*dt/plast.τ
  return nothing
end
