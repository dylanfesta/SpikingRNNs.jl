

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
  r1::Vector{Float64}
  r2::Vector{Float64}
  o1::Vector{Float64}
  o2::Vector{Float64}
  function PlasticityTriplets(τplus,τminus,τx,τy,
      A2plus,A3plus,A2minus,A3minus,n)
    new(τplus,τminus,τx,τy,A2plus,A3plus,A2minus,A3minus,
        ntuple(_->zeros(n),4)...)
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
     pspost::PSSpikingType,conn::Connection,pspre::PopulationState,plast::PlasticityTriplets)
  idx_pre_spike = findall(pspre.isfiring) 
  idx_post_spike = findall(pspost.isfiring) 
  # update synapses
	post_idxs = rowvals(conn.weights) # postsynaptic neurons
	weightsnz = nonzeros(conn.weights) # direct access to weights 
  # presynpatic spike go along w column
  for j in idx_pre_spike
		_posts_nz = nzrange(conn.weights,j) # indexes of corresponding pre in nz space
		@inbounds for _pnz in _posts_nz
      ipost = post_idxs[_pnz]
      weightsnz[_pnz]-=o1[ipost]*(plast.A2minus+plast.A3minus*r2[ipost])
    end
  end
  # postsynaptic spike: go along w row
  for i in idx_post_spike
    for _pnz in findall(==(i),post_idxs) # inefficient :-(
      w[_pnz]+=r1[i]*(plast.A2plus+plast.A3plus*o2[i])
    end
  end
  # update the plasticity trace variables
  for i in idx_pre_spike
    r1[i]+=1.0 ; r2[i]+=1.0
  end
  for i in idx_post_spike
    o1[i]+=1.0 ; o2[i]+=1.0
  end
  @. plast.r1 -= plast.r1*dt/plast.τplus
  @. plast.r2 -= plast.r2*dt/plast.τx
  @. plast.o1 -= plast.o1*dt/plast.τminux
  @. plast.o2 -= plast.o2*dt/plast.τy
  return nothing
end