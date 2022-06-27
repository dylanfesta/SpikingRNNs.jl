using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) #; plotlyjs();
using SparseArrays 
using SpikingRNNs; const global S = SpikingRNNs
using Test
using BenchmarkTools
using ProgressMeter
import FileIO: save # save raster png

using InvertedIndices

#using Random; Random.seed!(0)


function onesparsemat(w::Real)
  return sparse(cat(w;dims=2))
end

function wstick(wee::M,wie::M,wei::M,wii::M) where {R<:Real,M<:AbstractMatrix{R}}
  return Matrix(hcat(vcat(wee,wie), (-1).*abs.(vcat(wei,wii) )))
end

function rates_analytic(W::Matrix{R},h::Vector{R}) where R
  return (I-W)\h
end


##

using Plots ; theme(:dark)

##
# test 2, rate for 500 neurons should be the similar to 2D system 

ne = 350
ni = 150
N = ne+ni
dt = 0.1E-3
τe,τi = 0.5,0.2

he,hi = 70.,5.

sparse_ee = 0.3
sparse_ie = 0.3
sparse_ei = 0.4
sparse_ii = 0.4

τker_e = 0.8
τker_i = 0.4

wee_scal,wie_scal,wei_scal,wii_scal=(1.5,2.0,1.3,.8)

w_ee = S.sparse_constant_wmat(ne,ne,sparse_ee,1.0;rowsum=wee_scal) 
w_ie = S.sparse_constant_wmat(ni,ne,sparse_ie,1.0;rowsum=wie_scal,no_autapses=false) 
w_ei = S.sparse_constant_wmat(ne,ni,sparse_ei,1.0;rowsum=wei_scal,no_autapses=false) 
w_ii = S.sparse_constant_wmat(ni,ni,sparse_ii,1.0;rowsum=wii_scal) 

Wdense = wstick(w_ee,w_ie,w_ei,w_ii)
h_full = vcat(fill(he,ne),fill(hi,ni))

rats_an = rates_analytic(Wdense,h_full)

@info "expected E rate $(rats_an[1])"
@info "expected I rate $(rats_an[end])"


ps_e =  S.PSPoissonNeuron(τe,ne)
ps_i =  S.PSPoissonNeuron(τi,ni)

conn_ee = S.ConnectionPoissonExpKernel(S.PoissonExcitatory(),τker_e,w_ee)
conn_ie = S.ConnectionPoissonExpKernel(S.PoissonExcitatory(),τker_e,w_ie)
conn_ei = S.ConnectionPoissonExpKernel(S.PoissonInhibitory(),τker_i,w_ei)
conn_ii = S.ConnectionPoissonExpKernel(S.PoissonInhibitory(),τker_i,w_ii)

# inputs

in_e = S.PoissonInputCurrentConstant(fill(he,ne))
in_i = S.PoissonInputCurrentConstant(fill(hi,ni))

pop_e = S.Population(ps_e,
  (conn_ee,ps_e),(conn_ei,ps_i),(S.FakeConnection(),in_e))

pop_i = S.Population(ps_i,
  (conn_ie,ps_e),(conn_ii,ps_i),(S.FakeConnection(),in_i))


ntw = S.RecurrentNetwork(dt,pop_e,pop_i)

Ttot =20.0
# record spiketimes and internal potential
rec_spikes_e = S.RecSpikes(ps_e,800.0,Ttot)
rec_spikes_i = S.RecSpikes(ps_i,800.0,Ttot)

rec_state_e  = S.RecStateNow(ps_e,10,dt,Ttot;idx_save=[1,2,3])
rec_state_i  = S.RecStateNow(ps_i,10,dt,Ttot;idx_save=[1,2,3])

times = (0:ntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([ps_e,ps_i,rec_spikes_e,rec_spikes_i,rec_state_e,rec_state_i])
# initial conditions
ps_e.state_now .= 50.0
ps_i.state_now .= 50.0

@time begin
  @showprogress 5.0 "network simulation " for (k,t) in enumerate(times)
    rec_spikes_e(t,k,ntw)
    rec_spikes_i(t,k,ntw)
    rec_state_e(t,k,ntw)
    rec_state_i(t,k,ntw)
    S.dynamics_step!(t,ntw)
  end
end

spikec_e = S.get_content(rec_spikes_e)
spikec_i = S.get_content(rec_spikes_i)


rats_e = collect(values(S.get_mean_rates(spikec_e;Tstart=5.0)))
rats_i =collect(values( S.get_mean_rates(spikec_i;Tstart=5.0)))

histogram(values(rats_e) .- rats_an[1] )

@info "measured E rate $(mean(rats_e))"
@info "measured I rate $(mean(rats_i))"

myrast = S.draw_spike_raster( vcat(S.get_spiketrains(spikec_e)[1],S.get_spiketrains(spikec_i)[1]),
          0.001,1.0)
save("/tmp/rast.png",myrast)

@show rec_state_e.state_now[:,end]
@show rec_state_i.state_now[:,end]

scatter(rec_state_e.times,rec_state_e.state_now[3,:])
scatter!(rec_state_i.times,rec_state_i.state_now[3,:])
