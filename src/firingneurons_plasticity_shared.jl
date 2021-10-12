
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
  if !isempty(idx_post_spike)
    # innefficient ... need to search i element for each column
    for i_post in idx_post_spike
      for j_pre in (1:size(conn.weights,2))
        _pnz = searchsortedfirst(row_idxs,i_post,
            _colptr[j_pre],_colptr[j_pre+1]-1,Base.Order.Forward)
        if _pnz != _colptr[j_pre+1]  # update only if row i is present in row_idxs slice
          Δw = plast.r[j_pre]*plast.Aplus 
          weightsnz[_pnz]+= Δw
        end
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
  function PlasticityTriplets(τplus,τminus,τx,τy,
      A2plus,A3plus,A2minus,A3minus,n_post,n_pre)
    new(τplus,τminus,τx,τy,A2plus,A3plus,A2minus,A3minus,
        ntuple(_->zeros(n_post),2)...,
        ntuple(_->zeros(n_pre),2)...)
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
  # update synapses
  # presynpatic spike go along w column
  for j_pre in idx_pre_spike
		_posts_nz = nzrange(conn.weights,j_pre) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
      ipost = row_idxs[_pnz]
      Δw = plast.o1[ipost]*(plast.A2minus+plast.A3minus*plast.r2[j_pre])
      weightsnz[_pnz] = max(0.0,weightsnz[_pnz]-Δw)
    end
  end
  # postsynaptic spike: go along w row
  if !isempty(idx_post_spike)
    # innefficient ... need to search i element for each column
    for i_post in idx_post_spike
      for j_pre in (1:size(conn.weights,2))
        _pnz = searchsortedfirst(row_idxs,i_post,
            _colptr[j_pre],_colptr[j_pre+1]-1,Base.Order.Forward)
        if _pnz != _colptr[j_pre+1]  # update only if row i is present in row_idxs slice
          weightsnz[_pnz]+=plast.r1[j_pre]*(plast.A2plus+plast.A3plus*plast.o2[i_post])
        end
      end
    end
  end
  # update the plasticity trace variables
  for j_pre in idx_pre_spike
    plast.r1[j_pre]+=1.0 ; plast.r2[j_pre]+=1.0
  end
  for i_post in idx_post_spike
    plast.o1[i_post]+=1.0 ; plast.o2[i_post]+=1.0
  end
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
  function PlasticityInhibitoryVogels(τ,η,α,n_post,n_pre)
    new(τ,η,α,zeros(n_post),zeros(n_pre))
  end
end
function reset!(pl::PlasticityInhibitoryVogels)
  fill!(pl.r,0.0)
  fill!(pl.o,0.0)
  return nothing
end

function plasticity_update!(t_now::Real,dt::Real,
     pspost::PSSpikingType,conn::Connection,pspre::PSSpikingType,
     plast::PlasticityInhibitoryVogels)
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
      Δw = plast.η*(plast.o[ipost]-plast.α)
      weightsnz[_pnz] = min(0.0,weightsnz[_pnz]+Δw)
    end
  end
  # postsynaptic spike: go along w row
  if !isempty(idx_post_spike)
    # innefficient ... need to search i element for each column
    for j_pre in (1:size(conn.weights,2))
      for i_post in idx_post_spike
        _pnz = searchsortedfirst(row_idxs,i_post,
            _colptr[j_pre],_colptr[j_pre+1]-1,Base.Order.Forward)
        if _pnz != _colptr[j_pre+1]  # update only if row i is present in row_idxs slice
          Δw = plast.η*plast.r[j_pre]
          weightsnz[_pnz] = min(0.0,weightsnz[_pnz]+Δw)
        end
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