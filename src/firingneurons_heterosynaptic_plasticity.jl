

abstract type HeterosynapticConstraint end
abstract type HeterosynapticTarget end
abstract type HeterosynapticMethod end


struct HetUpperLimit <: HeterosynapticConstraint
  wsum_max::Float64
  wmin::Float64
  wmax::Float64
  tolerance::Float64
end
struct HetStrictSum <: HeterosynapticConstraint
  wsum_max::Float64
  wmin::Float64
  wmax::Float64
  tolerance::Float64
end

@inline function hardbounds(x::Float64,hc::HeterosynapticConstraint)
  hardbounds(x,hc.wmin,hc.wmax)
end

function _hetconstraint_vs_sum(sums::Vector{Float64},c::HetUpperLimit)
  _lim  =  c.wsum_max + c.tolerance
  return sums .> _lim
end
function _hetconstraint_vs_sum(sums::Vector{Float64},c::HetStrictSum)
  return .!isapprox.(sums,c.wsum_max;atol=c.tolerance)
end

struct HetIncoming <: HeterosynapticTarget end
struct HetOutgoing <: HeterosynapticTarget end
struct HetBoth <: HeterosynapticTarget end

struct HetAdditive <: HeterosynapticMethod end
struct HetMultiplicative <: HeterosynapticMethod end

struct PlasticityHeterosynapticSpikeTriggered{
    HC<:HeterosynapticConstraint,
    HM<:HeterosynapticMethod,
    HT<:HeterosynapticTarget} <:PlasticityRule 
  constraint::HC
  method::HM
  target::HT
  Δt_min_update::Float64
  _tlast::Ref{Float64}
end

function reset!(plast::PlasticityHeterosynapticSpikeTriggered)
  plast._tlast[] = -Inf
  return nothing
end

function plasticity_update!(t::R,::R,
     pspost::PopulationState,conn::Connection,::PopulationState,
     plast::PlasticityHeterosynapticSpikeTriggered) where R
  
  # to avoid excessively freqent updates
  if (t-plast._tlast[]) < plast.Δt_min_update  
    return nothing
  end
  # reset timer
  plast._tlast[] = t
  if !any(pspost.isfiring) # to optimize a little
    return nothing
  end
  idxs_fire = findall(pspost.isfiring)
  # check if must be normalized, returns sum and n elements
  idxs_change,sumvals,nvals = _hetplast_check(idxs_fire,
    conn.weights,
    plast.target,plast.constraint)
  # correct the weights only if needed
  if isempty(idxs_change)
    return nothing
  end
  # apply correction
  _apply_hetplast_spiketriggered!(idxs_change,sumvals,nvals,
      conn.weights,
      conn.target,conn.method,conn.constraint)
  return nothing
end

# incoming : check row value
# faster to go over all elements, than consider few ones :-(
function _hetplast_check(idxs::Vector{I},
    weights::SparseMatrixCSC{Float64,Int64},
    ::HetIncoming,constraint::HeterosynapticConstraint) where {I<:Integer}
  weights_nonzeros = nonzeros(weights)
  weights_rows = rowvals(weights)
  N = size(weights,1)
  n_el = fill(0,N)
  sums_all = fill(0.0,N) 
  for (_r,row) in enumerate(weights_rows)
    sums_all[row] += weights_nonzeros[_r]
    n_el[row] += 1
  end
  _sums_val = sums_all[idxs]
  _sums_nel = n_el[idxs]
  tochange = _hetconstraint_vs_sum(_sums_val,constraint)
  return idxs[tochange],_sums_val[tochange],_sums_nel[tochange]
end


# outgoing : column value
function _hetplast_check(idxs::Vector{I},
    weights::SparseMatrixCSC{Float64,Int64},
    ::HetOutgoing,constraint::HeterosynapticConstraint) where {I<:Integer}
  weights_nonzeros = nonzeros(weights)
  nsp=length(idxs)
  _sums_val = fill(NaN,nsp)
  _sums_nel = fill(-1,nsp)
  # now, sums over columns
  for (k,col) in enumerate(idxs)
    _range = nzrange(weights,col)
    lr = length(_range)
    if lr > 0
      _sums_val[k] = sum(view(weights_nonzeros,_range))
      _sums_nel[k] = length(_range)
    end
  end
  tochange = _hetconstraint_vs_sum(_sums_val,constraint)
  return idxs[tochange],_sums_val[tochange],_sums_nel[tochange]
end

# both incoming end outgoing. Returns two elements, one for 
# outgoing one for incoming
function _hetplast_check(idxs::Vector{I},
    weights::SparseMatrixCSC{Float64,Int64},
    ::HetBoth,constraint::HeterosynapticConstraint) where {
      I<:Integer }
  weights_nonzeros = nonzeros(weights)
  weights_rows = rowvals(weights)
  N = size(weights,1)
  n_el = fill(0,N)
  sums_all = fill(0.0,N) 
  for (_r,row) in enumerate(weights_rows)
    sums_all[row] += weights_nonzeros[_r]
    n_el[row] += 1
  end
  twoidxs = repeat(idxs;inner=2)
  twonsp=length(twoidxs)
  _sums_val = Vector{Float64}(undef,twonsp)
  _sums_nel = Vector{Int64}(undef,twonsp)
  # sum over rows : odd elements
  _sums_val[1:2:end] = sums_all[idxs]
  _sums_nel[1:2:end] = n_el[idxs]
  # now, sums over columns : even elements
  for (k,col) in enumerate(idxs)
    _range = nzrange(weights,col)
    lr = length(_range)
    if lr > 0
      _sums_val[2k] = sum(view(weights_nonzeros,_range))
      _sums_nel[2k] = length(_range)
    end
  end
  tochange = _hetconstraint_vs_sum(_sums_val,constraint)
  return twoidxs[tochange],_sums_val[tochange],_sums_nel[tochange]
end


# incoming additive (rows)
function _apply_hetplast_spiketriggered!(neus::Vector{I},
    sumvals::Vector{F},nelvals::Vector{I},
    weights::SparseMatrixCSC{Float64,Int64},
    ::HetIncoming,
    ::HetAdditive,constraint::HeterosynapticConstraint) where {
      I<:Integer,F<:Real}
  weights_nonzeros = nonzeros(weights)
  weights_rows = rowvals(weights)
  N = size(weights,1)
  # feels like MATLAB !!!
  fix_vals_all = fill(0.0,N) # much empty!
  for (k,neu) in enumerate(neus)
    fix_vals_all[neu] = (constraint.wsum_max - sumvals[k]) ./ nelvals[k]
  end
  for (_r,row) in enumerate(weights_rows)
    _fix = fix_vals_all[row]
    if !iszero(_fix)
    weights_nonzeros[_r] = hardbounds(
          weights_nonzeros[_r]+_fix,constraint)
    end
  end
  return nothing
end  

# outgoing additive (columns)
function _apply_hetplast_spiketriggered!(neus::Vector{I},
    sumvals::Vector{F},nelvals::Vector{I},
    weights::SparseMatrixCSC{Float64,Int64},
    ::HetOutgoing,
    ::HetAdditive,constraint::HeterosynapticConstraint) where {
      I<:Integer,F<:Real}
  # pass over columns
  weights_nonzeros = nonzeros(weights)
  for (k,col) in enumerate(neus)
    fixval_k = (constraint.wsum_max - sumvals[k]) / nelvals[k]
    for _r in nzrange(weights,col)
      weights_nonzeros[_r] = hardbounds(
        weights_nonzeros[_r]+fixval_k,constraint)
    end
  end
  return nothing
end  

# both outoing and incoming additive (rows and columns)
function _apply_hetplast_spiketriggered!(neus::Vector{I},
    sumvals::Vector{F},nelvals::Vector{I},
    weights::SparseMatrixCSC{Float64,Int64},
    ::HetBoth,
    ::HetAdditive,constraint::HeterosynapticConstraint) where {
      I<:Integer,F<:Real}
  # row part , odd elements
  weights_nonzeros = nonzeros(weights)
  weights_rows = rowvals(weights)
  N = size(weights,1)
  fix_vals_all = fill(0.0,N)
  for k in 1:2:length(neus)
    row = neus[k]
    fix_vals_all[row] = (constraint.wsum_max - sumvals[k]) ./ nelvals[k]
  end
  for (_r,row) in enumerate(weights_rows)
    _fix = fix_vals_all[row]
    if !iszero(_fix)
      weights_nonzeros[_r] = hardbounds(
          weights_nonzeros[_r]+_fix,constraint)
    end
  end
  # now, pass over columns : even elements
  for k in 2:2:length(neus)
    col = neus[k]
    fix_val_k = (constraint.wsum_max - sumvals[k]) / nelvals[k]
    for _r in nzrange(weights,col)
      weights_nonzeros[_r] = hardbounds(
        weights_nonzeros[_r]+fix_val_k,constraint)
    end
  end
  return nothing
end  


# and here is the EASY plasticity, instead of spike triggered
# ... to rewrite to make it more uniform to the above 

abstract type HeterosynapticPlasticityMethod end
struct HeterosynapticAdditive <: HeterosynapticPlasticityMethod
  wmin::Float64
  wmax::Float64
end
struct HeterosynapticMultiplicative <: HeterosynapticPlasticityMethod 
  wmin::Float64
  wmax::Float64
end

abstract type HeterosynapticPlasticityTarget end
struct HeterosynapticIncoming <: HeterosynapticPlasticityTarget 
  sum_max::Float64
end 
struct HeterosynapticOutgoing <: HeterosynapticPlasticityTarget 
  sum_max::Float64
end 

struct PlasticityHeterosynapticEasy{HetMeth<:HeterosynapticPlasticityMethod,
    HetTarg<:HeterosynapticPlasticityTarget} <: PlasticityRule
  Δt_update::Float64
  method::HetMeth
  target::HetTarg
  _tcounter::Ref{Float64}
  function PlasticityHeterosynapticEasy(Δt_update::Float64,
      method::M,target::T) where {M<:HeterosynapticPlasticityMethod,T<:HeterosynapticPlasticityTarget}
    counter = Ref(0.0)
    new{M,T}(Δt_update,method,target,counter)
  end
end

function reset!(plast::PlasticityHeterosynapticEasy)
  plast._tcounter[] = zero(Float64)
  return nothing
end

function plasticity_update!(::R,dt::R,
     pspost::PopulationState,conn::Connection,pspre::PopulationState,
     plast::PlasticityHeterosynapticEasy) where R
  if plast._tcounter[] < plast.Δt_update  
    plast._tcounter[] += dt
    return nothing
  end
  # reset timer
  plast._tcounter[] = zero(R)
  # apply plasticity at each col/row , depending on target
  _apply_easy_het_plasticity!(conn.weights,plast.method,plast.target)
  return nothing
end


@inline function hardbounds(x::Real,m::HeterosynapticPlasticityMethod)
  return  hardbounds(x,m.wmin,m.wmax)
end

# this should be a little faster than sum(M;dims=2)
function sum_over_rows!(dest::Vector,M::SparseMatrixCSC)
  _rowidx = SparseArrays.rowvals(M)
  Mnz = nonzeros(M) # direct access to weights 
  @inbounds for (i,r) in enumerate(_rowidx)
    dest[r] += Mnz[i]
  end
  return dest
end

# how many elements are present along each col
function count_col_elements(M::SparseMatrixCSC)
  ncols = size(M,2)
  return length.(nzrange.(Ref(M),1:ncols))
end

# how many elements are present along each row
function count_row_elements(M::SparseMatrixCSC)
  nrows = size(M,1)
  ret = fill(0,nrows)
  for r in rowvals(M)
    ret[r]+=1
  end
  return ret
end

# additive, incoming connections corresponds to rows
function _apply_easy_het_plasticity!(sparseweights::SparseMatrixCSC,
    method::HeterosynapticAdditive,
    target::HeterosynapticIncoming)
  sum_max = target.sum_max
  nzw = nonzeros(sparseweights)
  wrows = rowvals(sparseweights)
  fixthing = Vector{Float64}(undef,size(sparseweights,1))
  sum_over_rows!(fixthing,sparseweights)
  n_el_row = count_row_elements(sparseweights)
  for i in eachindex(fixthing)
    _sum = fixthing[i]
    if _sum > sum_max
      fixthing[i] = (sum_max-_sum)/n_el_row[i]
    else
      fixthing[i] = 0.0
    end
  end
  for (_r,row) in enumerate(wrows)
    fix = fixthing[row]
    if !iszero(fix) # faster ?
      nzw[_r] = hardbounds(nzw[_r]+fix,method)
    end
  end
  return nothing
end

# additive, outgoing connections corresponds to columns
function _apply_easy_het_plasticity!(sparseweights::SparseMatrixCSC,
    method::HeterosynapticAdditive,
    target::HeterosynapticOutgoing)
  sum_max = target.sum_max
  nzw = nonzeros(sparseweights)
  sumswee = sum(sparseweights;dims=1)
  for (c,_sumr) in enumerate(sumswee)
    if _sumr > sum_max
      idxs = nzrange(sparseweights,c)
      for idx in idxs
        newval = nzw[idx] - (sumswee[c] - sum_max)/length(idxs)
        nzw[idx] = hardbounds(newval,method)
      end
    end
  end
  return nothing
end


# "precise" implementation... BUT... 
# this code is just too slow :-( 
struct PlasticityHeterosynaptic{HetMeth<:HeterosynapticPlasticityMethod,
    HetTarg<:HeterosynapticPlasticityTarget} <: PlasticityRule
  Δt_update::Float64
  method::HetMeth
  target::HetTarg
  _tcounter::Ref{Float64}
  function PlasticityHeterosynaptic(Δt_update::Float64,
      method::M,target::T) where {M<:HeterosynapticPlasticityMethod,T<:HeterosynapticPlasticityTarget}
    counter = Ref(0.0)
    new{M,T}(Δt_update,method,target,counter)
  end
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


function plasticity_update!(::Real,dt::Real,
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