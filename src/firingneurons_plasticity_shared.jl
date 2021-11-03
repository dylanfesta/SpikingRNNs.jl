
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
@inline function plasticity_update!(t_now::Real,dt::Real,
     pspost::PSSpikingType,conn::Connection,pspre::PSSpikingType,
     plast::NoPlasticity)
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
  function PairSTDP(τplus,τminus,
      Aplus,Aminus,n_post,n_pre)
    new(τplus,τminus,Aplus,Aminus,
      zeros(n_post),zeros(n_pre))
  end
end

function reset!(pl::PairSTDP)
  fill!(pl.r,0.0)
  fill!(pl.o,0.0)
  return nothing
end

function plasticity_update!(t_now::Real,dt::Real,
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
      Δw = plast.o[ipost]*plast.Aminus
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

function plasticity_update!(t_now::Real,dt::Real,
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

# simple structural plasticity

struct PlasticityStructural <: PlasticityRule
  Δt_update::Float64
  w_new::Float64  
  νdeath::Float64
  νbirth::Float64
  w_density::Float64
  no_autapses::Bool
  _tcounter::Ref{Float64}
end

function reset!(plast::PlasticityStructural)
  plast._tcounter[] = zero(Float64)
  return nothing
end

# ρ is connection density , w_density is ration between n_connections/ n_non_connections
function PlasticityStructural(νdeath::R,Δt_update::R,ρ::R,w_new::R;
    no_autapses::Bool=true) where R<:Float64
  w_density = ρ/(1-ρ)  
  @assert w_density > 0.0
  νbirth = w_density*νdeath
  @assert νbirth*Δt_update < 1.0
  _tcounter = Ref(zero(Float64))
  return PlasticityStructural(Δt_update,w_new,νdeath,νbirth,w_density,no_autapses,
   _tcounter)
end

function plasticity_update!(t_now::Real,dt::Real,
     pspost::PopulationState,conn::Connection,pspre::PopulationState,
     plast::PlasticityStructural)
  plast._tcounter[] += dt
  if plast._tcounter[] < plast.Δt_update  
    return nothing
  end
  # reset timer
  plast._tcounter[] = zero(Float64)
  # How many to update 
  n_alive = nnz(conn.weights)
  n_tot = (*)(size(conn.weights)...)
  if plast.no_autapses
    n_tot -= min(size(conn.weights)...)
  end
  n_unborn = n_tot - n_alive
  nbirth = ceil(Integer,plast.νbirth * plast.Δt_update * n_unborn)
  ndeath = ceil(Integer,plast.νbirth * plast.Δt_update * n_alive)
  # the updates are slow on large sparse matrices :-(
  for _ in 1:nbirth
    newij = sample(CartesianIndices(conn.weights))
    while !( (conn.weights[newij] == 0.0) || 
           (plast.no_autapses && newij[1]==newij[2]) )
      newij = sample(CartesianIndices(conn.weights))
    end
    conn.weights[newij] = plast.w_new
  end
  for _ in 1:ndeath
    idx_kill = sample(1:nnz(conn.weights))
    nonzeros(conn.weights)[idx_kill] = 0.0
  end
  dropzeros!(conn.weights)
  return nothing
end

# Heterosynaptic plasticity modes

abstract type HeterosynapticPlasticityMethod end
struct HeterosynapticAdditive <: HeterosynapticPlasticityMethod
  wmin::Float64
end
struct HeterosynapticMultiplicative <: HeterosynapticPlasticityMethod 
  wmin::Float64
end

abstract type HeterosynapticPlasticityTarget end
struct HeterosynapticIncoming <: HeterosynapticPlasticityTarget 
  sum_max::Float64
end 
struct HeterosynapticOutgoing <: HeterosynapticPlasticityTarget 
  sum_max::Float64
end 

struct PlasticityHeterosynaptic{HetMeth<:HeterosynapticPlasticityMethod,HetTarg<:HeterosynapticPlasticityTarget} <: PlasticityRule
  method::HetMeth
  target::HetTarg
  _tcounter::Ref{Float64}
end

function reset!(plast::PlasticityHeterosynaptic)
  plast._tcounter[] = zero(Float64)
  return nothing
end

function _get_row_idxs(M::SparseMatrixCSC,row_idx::Integer)
  return findall(==(row_idx), SparseArrays.rowvals(M))
end
function _get_col_idxs(M::SparseMatrixCSC,col_idx::Integer)
  cptr = SparseArrays.getcolptr(M)
  return collect(cptr[col_idx]:cptr[col_idx+1]-1)
end

# let's have an iterator to handle those two above, for each row/col
struct HeterosynapticIdxsIterator{HetTarg}
  M::SparseMatrixCSC
  target::HetTarg
end
Base.eltype(::Type{HeterosynapticIdxsIterator}) = Vector{Int64}
Base.getindex(hetit::HeterosynapticIdxsIterator,idx::Integer) = iterate(hetit,idx)[1]

# incoming connections: I am iterating over rows
Base.length(hetit::HeterosynapticIdxsIterator{HeterosynapticIncoming}) = size(hetit.M,1)
function Base.iterate(hetit::HeterosynapticIdxsIterator{HeterosynapticIncoming})
  idxs = _get_row_idxs(hetit.M,1)
  return (idxs,2)
end
function Base.iterate(hetit::HeterosynapticIdxsIterator{HeterosynapticIncoming},
    row::Int64)
  if row > length(hetit)
    return nothing
  else
    idxs = _get_row_idxs(hetit.M,row)
    row += 1
    return (idxs,row)
  end
end
# outgoing connections : iterate over columns
Base.length(hetit::HeterosynapticIdxsIterator{HeterosynapticOutgoing}) = size(hetit.M,2)
function Base.iterate(hetit::HeterosynapticIdxsIterator{HeterosynapticOutgoing})
  idxs = _get_col_idxs(hetit.M,1)
  return (idxs,2)
end
function Base.iterate(hetit::HeterosynapticIdxsIterator{HeterosynapticOutgoing},
    row::Int64)
  if row > length(hetit)
    return nothing
  else
    idxs = _get_col_idxs(hetit.M,row)
    row += 1
    return (idxs,row)
  end
end


# if values less than wmin, re-distributes the correction on the other neurons
function _heterosynaptic_fix!(nzvals::Vector{Float64},idxs::Vector{<:Integer},
    method::HeterosynapticAdditive,target::HeterosynapticPlasticityTarget)
  wmin = method.wmin
  nsyn = length(idxs)
  # fix without wmin
  fix_val = (sum(view(nzvals,idxs))-target.sum_max)/nsyn
  (fix_val < 0.0) && (return nothing)
  while true
    docycle = false
    accu = 0.0
    idx_low = Int64[]
    for idx in idxs
      nzvals[idx] -= fix_val
      if nzvals[idx] < wmin
        accu += wmin - nzvals[idx]
        nzvals[idx] = wmin
        push!(idx_low,idx)
        docycle=true
      end
    end
    if docycle
      filter!(!(in)(idx_low),idxs)
      fix_val = accu / (nsyn-length(idx_low))
      nsyn = length(idxs)
    else
      break
    end
  end
  return nothing
end

function _heterosynaptic_fix!(nzvals::Vector{Float64},idxs::Vector{<:Integer},
    method::HeterosynapticMultiplicative,target::HeterosynapticPlasticityTarget)
  wmin = method.wmin
  sum_targ = target.sum_max
  sum_val = sum(view(nzvals,idxs))
  fix_val = sum_targ/sum_val 
  (fix_val > 1.0) && (return nothing)
  while true
    docycle = false
    sum_val = 0.0
    idx_low = Int64[]
    for idx in idxs
      nzvals[idx] *= fix_val # apply fix
      if nzvals[idx] < wmin  # check the fix
        nzvals[idx] = wmin
        push!(idx_low,idx)
        docycle = true
        sum_targ -= wmin  # update target sum to exclude min synapses
      else
        sum_val += nzvals[idx]  # update synapses sum
      end
    end
    if docycle
      filter!(!(in)(idx_low),idxs)
      fix_val = sum_targ/sum_val 
    else
      break
    end
  end
  return nothing
end


function plasticity_update!(t_now::Real,dt::Real,
     pspost::PopulationState,conn::Connection,pspre::PopulationState,
     plast::PlasticityHeterosynaptic)
  plast._tcounter[] += dt
  if plast._tcounter[] < plast.Δt_update  
    return nothing
  end
  # reset timer
  plast._tcounter[] = zero(Float64)
  # apply plasticity at each col/row , depending on target
  wnzvals = nonzeros(conn.weights)
  for idxs in HeterosynapticIdxsIterator(conn.weights,plast.target)
    _heterosynaptic_fix!(wnzvals,idxs,plast.method,plast.target)
  end
  return nothing
end