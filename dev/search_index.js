var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SpikingRNNs","category":"page"},{"location":"#SpikingRNNs","page":"Home","title":"SpikingRNNs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SpikingRNNs.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SpikingRNNs]","category":"page"},{"location":"#SpikingRNNs.forward_signal!-Tuple{Real, Real, SpikingRNNs.PSRate, SpikingRNNs.BaseConnection, SpikingRNNs.PSRate}","page":"Home","title":"SpikingRNNs.forward_signal!","text":"send_signal(conn::ConnectionStateRate)\n\nComputes the input to postsynaptic population, given the current state of presynaptic population. For a rate model, it applies the iofunction to the neuron potentials, gets the rate values then multiplies rates by weights, adding the result to the input of the postsynaptic population.\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.hardbounds-Union{Tuple{R}, Tuple{R, R, R}} where R","page":"Home","title":"SpikingRNNs.hardbounds","text":"hardbounds(x::R,low::R,high::R) where R = min(high,max(x,low))\n\nApplies hard-bounds on scalar x  \n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.lognorm_reparametrize-Tuple{Real, Real}","page":"Home","title":"SpikingRNNs.lognorm_reparametrize","text":"lognorm_reparametrize(m::Real,std::Real) -> d::LogNormal\n\nParameters\n\nm::Real   sample mean\nstd::Real sample std\n\nReturns\n\nd::Distributions.LogNormal\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.next_poisson_spiketime-Tuple{Float64, Float64}","page":"Home","title":"SpikingRNNs.next_poisson_spiketime","text":"nextpoissonspiketime(tcurrent::Float64,rate::Float64) -> tnext::Float64\n\nReturns next spike after current time t_current in a random Poisson process.   with rate rate.\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.next_poisson_spiketime_from_function-Tuple{Float64, Float64, Float64}","page":"Home","title":"SpikingRNNs.next_poisson_spiketime_from_function","text":"nextpoissonspiketimefromfunction(tcurrent::Float64,funrate::Function,funrateupper::Function;        Tmax::Float64=0.0,nowarning::Bool=false) -> Float64\n\nReturns the next spiketime in a Poisson process with time-varying rate. The rate variation is given by function fun_rate.\n\nSee e.g.  Laub,Taimre,Pollet 2015\n\nArguments\n\nt_current::Float64 : current time \nfun_rate::Function : fun_rate(t::Float64) -> r::Float64 returns rate at time t \nfun_rate_upper::Function : upper limit to the function above. Strictly decreasing in t  must be as close as possible to the fun_rate for efficiency\nTmax::Float64 : upper threshold for spike proposal, maximum interval that can be produced    \nnowarning::Bool : does not throw a warning when Tmax` is reached\n\n\n\n\n\n","category":"method"}]
}
