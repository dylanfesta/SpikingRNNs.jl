

abstract type PlasticityStructuralType end
abstract type PlasticityStructuralSynapticGeneration end


struct SynapticGenerationConstant <: PlasticityStructuralSynapticGeneration
  birth_weight::Float64
end

function (gen::SynapticGenerationConstant)()
  return gen.birth_weight
end


struct StructuralPlasticityPlain <: PlasticityStructuralType
  νdeath::Float64
  νbirth::Float64
end

function StructuralPlasticityPlain(;
    connection_density::R=0.2,death_rate::R=1.0) where R
  w_density = connection_density/(1-connection_density)  
  birth_rate = death_rate * w_density
  return StructuralPlasticityPlain(death_rate,birth_rate)
end

struct StructuralPlasticityWeightDependent <:  PlasticityStructuralType
  νdeath::Float64
  νbirth::Float64
  w_temperature::Float64
end

function StructuralPlasticityWeightDependent(;
    connection_density::R=0.2,death_rate::R=1.0,
    w_temperature::R=1.0) where R
  w_density = connection_density/(1-connection_density)  
  birth_rate = death_rate * w_density
  return StructuralPlasticityWeightDependent(
          death_rate,birth_rate,w_temperature)
end

struct PlasticityStructural{
    PST<:PlasticityStructuralType,
    PSSyngen <: PlasticityStructuralSynapticGeneration,
      } <: PlasticityRule
  Δt_update::Float64
  structural_type::PST
  synaptic_generation::PSSyngen
  no_autapses::Bool
  _tcounter::Ref{Float64}
  function PlasticityStructural(
      pstype::PST,syngen::PSSynGen,
      Δt::Float64;
      no_autapses=true
      ) where {PST<:PlasticityStructuralType,
                PSSynGen<:PlasticityStructuralSynapticGeneration}
    return new{PST,PSSynGen}(Δt,pstype,syngen,no_autapses,Ref(0.0))  
  end

end

function reset!(plast::PlasticityStructural)
  plast._tcounter[] = zero(Float64)
  return nothing
end

function _add_new_random_synapses!(weights::SparseMatrixCSC,nadd::Integer,
    syngen::PlasticityStructuralSynapticGeneration,no_autapses::Bool)
  (nadd == 0) && (return nothing) # meh
  wextend = ExtendableSparseMatrix(weights)  
  nbirthed = 0
  widxs_all = CartesianIndices(weights)
  while nbirthed <= nadd
    (i,j) = Tuple(sample(widxs_all))
    if  iszero(ExtendableSparse.findindex(weights,i,j)) && (
        !((i==j) && no_autapses) )
      neww = syngen()
      setindex!(wextend,neww,i,j)
      nbirthed +=1
    end
  end
  flush!(wextend)
  copy!(weights,wextend.cscmatrix)
  dropzeros!(weights)
  return nothing
end

function _get_nbirth(Δt::Real,weights::SparseMatrixCSC,no_autapses::Bool,
      strtype::Union{
            StructuralPlasticityPlain,
            StructuralPlasticityWeightDependent})
  Ntot = (*)(size(weights)...)
  if no_autapses
    Ntot -= min(size(weights)...)
  end
  Nunconnected = Ntot - nnz(weights)           
  return round(Integer,
    (1-exp(-strtype.νbirth * Δt)) * Nunconnected)
end
function _get_ndeath(Δt::Real,weights::SparseMatrixCSC,strtype::Union{
            StructuralPlasticityPlain,
            StructuralPlasticityWeightDependent})
  return round(Integer,
    (1-exp(-strtype.νdeath * Δt)) * nnz(weights))
end

function plasticity_update!(::Real,dt::Real,
     ::PopulationState,conn::Connection,::PopulationState,
     plast::PlasticityStructural{
        StructuralPlasticityPlain,
        <:PlasticityStructuralSynapticGeneration})
  plast._tcounter[] += dt
  if plast._tcounter[] < plast.Δt_update  
    return nothing
  end
  # reset timer
  plast._tcounter[] = zero(Float64)
  weights = conn.weights
  # death first!
  ndeath = _get_ndeath(plast.Δt_update, weights,plast.structural_type)
  weightsnz = nonzeros(weights)
  Nsynapses = nnz(weights)
  idx_kill = sample(1:Nsynapses,ndeath;replace=false)
  weightsnz[idx_kill] .= 0.0
  dropzeros!(weights)
  # Now birth  
  nbirth = _get_nbirth(plast.Δt_update, weights, plast.no_autapses,
    plast.structural_type)
  _add_new_random_synapses!(weights,nbirth,plast.synaptic_generation,
    plast.no_autapses)
  return nothing
end


# weight dependent structural plasticity!

function plasticity_update!(::Real,dt::Real,
     ::PopulationState,conn::Connection,::PopulationState,
     plast::PlasticityStructural{
        StructuralPlasticityWeightDependent,
        <:PlasticityStructuralSynapticGeneration})
  plast._tcounter[] += dt
  if plast._tcounter[] < plast.Δt_update  
    return nothing
  end
  # reset timer
  plast._tcounter[] = zero(Float64)
  # How many to update 
  weights = conn.weights
  weightsnz = nonzeros(weights)
  # death first, do several swipes if necessary
  wnonzeros = nonzeros(weights)
  pkill = @. exp(-wnonzeros/plast.structural_type.w_temperature)
  pkill ./= sum(pkill)
  ndeath = _get_ndeath(plast.Δt_update, weights,plast.structural_type)
  nkilled = 0
  while nkilled < ndeath
    idx_kill = unique(rand(Categorical(pkill),ndeath-nkilled))
    weightsnz[idx_kill] .= 0.0 
    pkill .*= 1/(1-sum(pkill[idx_kill]))
    pkill[idx_kill] .= 0.0
    nkilled += length(idx_kill)
  end
  dropzeros!(weights)
  # Now birth, same as plain type
  nbirth = _get_nbirth(plast.Δt_update, weights, plast.no_autapses,
    plast.structural_type)
  _add_new_random_synapses!(weights,nbirth,plast.synaptic_generation,
    plast.no_autapses)
  return nothing
end