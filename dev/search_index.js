var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SpikingRNNs","category":"page"},{"location":"#SpikingRNNs","page":"Home","title":"SpikingRNNs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"warning: Warning\nThe documentation is still missing. Please see the \"examples\" section for usage.","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Leaky-integrate and fire, single neurons – Voltage traces, refractoriness, etc.\nLeaky-integrate and fire, sinusoidal input – Voltage traces, refractoriness, etc.","category":"page"},{"location":"#Index-and-documented-functions","page":"Home","title":"Index and documented functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SpikingRNNs]","category":"page"},{"location":"#SpikingRNNs.PSIFNeuron-Union{Tuple{R}, Tuple{Integer, R, R, R, R, R, R}} where R","page":"Home","title":"SpikingRNNs.PSIFNeuron","text":"PSIFNeuron(n::Integer,τ::R,cap::R,\n    v_threshold::R,v_reset::R,v_leak::R,t_refractory::R) where R\n\nGenerates a population of leaky integrate and fire neurons.\n\nArguments\n\nn : number of neurons\nτ : dynamics time constant\ncap : capacitance\nv_threshold : treshold for action potential emission\nv_reset : reset value after firing\nv_leak : leak potential\nt_refractory : duration of refractory period\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.bin_spikes-Union{Tuple{R}, Tuple{Vector{R}, R, R}} where R","page":"Home","title":"SpikingRNNs.bin_spikes","text":"bin_spikes(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R\n\nArguments\n\nY::Vector{<:Real} : vector of spike times\ndt::Real : time bin size\nTend::Real : end time for the raster\n\nOptional argument\n\nTstart::Real=0.0 : start time for the raster\n\nReturns\n\nbinned_spikes::Vector{<:Integer} : binned_spikes[k] is the number of spikes that occur    in the timebin k  (i.e. between Tstart + (k-1)*dt and Tstart + k*dt)\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.draw_spike_raster-Union{Tuple{C}, Tuple{Vector{Vector{Float64}}, Real, Real}} where C<:ColorTypes.Color","page":"Home","title":"SpikingRNNs.draw_spike_raster","text":"drawspikeraster(trains::Vector{Vector{Float64}},       dt::Real,Tend::Real;       Tstart::Real=0.0,       spikesize::Integer = 5,       spikeseparator::Integer = 1,       backgroundcolor::Color=RGB(1.,1.,1.),       spikecolors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),       max_size::Real=1E4) where C<:Color\n\nDraws a matrix that contains the raster plot of the spike train.\n\nArguments\n\nTrains :  Vector of spike trains. The order of the vector corresponds to  the order of the plot. First element is at the top, second is second row, etc.\ndt : time interval representing one horizontal pixel  \nTend : final time to be considered\n\nOptional arguments\n\nTstart::Real : starting time\nspike_size::Integer : heigh of spike (in pixels)\nspike_separator::Integer : space between spikes, and vertical padding\nbackground_color::Color : self-explanatory\nspike_colors::Union{Color,Vector{Color}} : if a single color, color of all spikes, if vector of colors,   color for each neuron (length should be same as number of neurons)\nmax_size::Integer : throws an error if image is larger than this number (in pixels)\n\nReturns\n\nraster_matrix::Matrix{Color} you can save it as a png file\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.forward_signal!-Tuple{Real, Real, SpikingRNNs.PSRate, SpikingRNNs.BaseConnection, SpikingRNNs.PSRate}","page":"Home","title":"SpikingRNNs.forward_signal!","text":"send_signal(conn::ConnectionStateRate)\n\nComputes the input to postsynaptic population, given the current state of presynaptic population. For a rate model, it applies the iofunction to the neuron potentials, gets the rate values then multiplies rates by weights, adding the result to the input of the postsynaptic population.\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.hardbounds-Union{Tuple{R}, Tuple{R, R, R}} where R","page":"Home","title":"SpikingRNNs.hardbounds","text":"hardbounds(x::R,low::R,high::R) where R = min(high,max(x,low))\n\nApplies hard-bounds on scalar x  \n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.lognorm_reparametrize-Tuple{Real, Real}","page":"Home","title":"SpikingRNNs.lognorm_reparametrize","text":"lognorm_reparametrize(m::Real,std::Real) -> d::LogNormal\n\nParameters\n\nm::Real   sample mean\nstd::Real sample std\n\nReturns\n\nd::Distributions.LogNormal\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.next_poisson_spiketime-Tuple{Float64, Float64}","page":"Home","title":"SpikingRNNs.next_poisson_spiketime","text":"nextpoissonspiketime(tcurrent::Float64,rate::Float64) -> tnext::Float64\n\nReturns next spike after current time t_current in a random Poisson process.   with rate rate.\n\n\n\n\n\n","category":"method"},{"location":"#SpikingRNNs.next_poisson_spiketime_from_function-Tuple{Float64, Function, Function}","page":"Home","title":"SpikingRNNs.next_poisson_spiketime_from_function","text":"nextpoissonspiketimefromfunction(tcurrent::Float64,funrate::Function,funrateupper::Function;        Tmax::Float64=0.0,nowarning::Bool=false) -> Float64\n\nReturns the next spiketime in a Poisson process with time-varying rate. The rate variation is given by function fun_rate.\n\nSee e.g.  Laub,Taimre,Pollet 2015\n\nArguments\n\nt_start::Float64 : current time \nfun_rate::Function : fun_rate(t::Float64) -> r::Float64 returns rate at time t \nfun_rate_upper::Function : upper limit to the function above. Strictly decreasing in t  must be as close as possible to the fun_rate for efficiency\nTmax::Float64 : upper threshold for spike proposal, maximum interval that can be produced    \nnowarning::Bool : does not throw a warning when Tmax` is reached\n\n\n\n\n\n","category":"method"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"EditURL = \"https://github.com/dylanfesta/SpikingRNNs.jl/blob/master/examples/if_modulated_input.jl\"","category":"page"},{"location":"if_modulated_input/#Sinuisodal-input-for-two-LIF-neurons","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"","category":"section"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"Several LIF E neurons receive a sinusoidal spiking input.","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"I plot the input and the spike train.","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"push!(LOAD_PATH, abspath(@__DIR__,\"..\"))\n\nusing Test\nusing LinearAlgebra,Statistics,StatsBase,Distributions\nusing Plots,NamedColors ; theme(:default)\nusing SparseArrays\nusing SpikingRNNs; const global S = SpikingRNNs\nusing FileIO\n\n# # src","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"Time","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const dt = 1E-3\nconst Ttot = 5.0;\nnothing #hide","category":"page"},{"location":"if_modulated_input/#Create-10-excitatory-LIF-neurons","page":"Sinuisodal input for two LIF neurons","title":"Create 10  excitatory LIF neurons","text":"","category":"section"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const N = 10\nconst τ = 0.2 # time constant for dynamics\nconst cap = τ # capacitance\nconst vth = 10.  # action-potential threshold\nconst vreset = -5.0 # reset potential\nconst vleak = -5.0 # leak potential\nconst τrefr = 0.0 # refractoriness\nconst τpcd = 0.02 # synaptic kernel decay\n\nconst ps = S.PSIFNeuron(N,τ,cap,vth,vreset,vleak,τrefr);\nnothing #hide","category":"page"},{"location":"if_modulated_input/#Modulation-signal","page":"Sinuisodal input for two LIF neurons","title":"Modulation signal","text":"","category":"section"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const ω = 1.0\nconst text_start = 0.23 # when the signal is on\nconst rext_min,rext_max = 10,200\nconst rext_off = 6.0\n\nfunction ratefun(t::Float64)\n  (t<=text_start) && (return rext_off)\n  return rext_min + (0.5+0.5sin(2π*t/ω))*(rext_max-rext_min)\nend\nfunction ratefun_upper(::Float64)\n  return rext_max # slightly suboptimal when it comes to generation of spikes\nend","category":"page"},{"location":"if_modulated_input/#Input-object","page":"Sinuisodal input for two LIF neurons","title":"Input object","text":"","category":"section"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const ps_input = S.IFInputSpikesFunScalar(N,ratefun,ratefun_upper)","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"connection from input to E let's define a conductance based kernel this time!","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const τker = 0.3\nconst vrev_in = 15.0 # must be higher than firing threshold!\nconst in_ker = S.SyKConductanceExponential(N,τker,vrev_in)\nconst win = 0.1\nconst in_weights = fill(win,N)\nconst conn_e_in = S.ConnectionIFInput(in_weights,in_ker)","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"Now I can define the population and the network. The neurons have no mutual connections, they are independent","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const pop = S.Population(ps,(conn_e_in,ps_input))\nconst network = S.RecurrentNetwork(dt,pop)\n\n\n# I will record the full spike train for the neurons.\nconst rec_spikes = S.RecSpikes(ps,50.0,Ttot)","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"and the internal potential","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const krec = 1\nconst rec_state = S.RecStateNow(ps,krec,dt,Ttot)\n\nconst times = (0:dt:Ttot)\nconst nt = length(times);\nnothing #hide","category":"page"},{"location":"if_modulated_input/#Run-the-network","page":"Sinuisodal input for two LIF neurons","title":"Run the network","text":"","category":"section"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"S.reset!.([rec_state,rec_spikes])\nS.reset!(ps);\nfill!(ps.state_now,0.0)\n\nfor (k,t) in enumerate(times)\n  rec_state(t,k,network)\n  rec_spikes(t,k,network)\n  S.dynamics_step!(t,network)\nend","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"this is useful for visualization only","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"S.add_fake_spikes!(1.5vth,rec_state,rec_spikes)","category":"page"},{"location":"if_modulated_input/#Plot-internal-potential-for-a-pair-of-neurons","page":"Sinuisodal input for two LIF neurons","title":"Plot internal potential for a pair of neurons","text":"","category":"section"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"_ = let neu1=1,neu2=2,\n  times = rec_state.times,\n  mpot1 = rec_state.state_now[neu1,:]\n  mpot2 = rec_state.state_now[neu2,:]\n  plot(times,[mpot1 mpot2],\n    xlabel=\"time (s)\",\n    ylabel=\"membrane potential (mV)\",\n    label=[\"neuron 1\" \"neuron 2\"],\n    leg=:bottomright)\nend","category":"page"},{"location":"if_modulated_input/#Plot-train-raster","page":"Sinuisodal input for two LIF neurons","title":"Plot train raster","text":"","category":"section"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"const trains = S.get_spiketrains(rec_spikes)\n\ntheraster = let rdt = 0.01,\n  rTend = Ttot\n  S.draw_spike_raster(trains,rdt,rTend)\nend","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"The raster might not be visible online, but it can be saved locally as a png image as follows: save(\"<save path>\",theraster)","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"","category":"page"},{"location":"if_modulated_input/","page":"Sinuisodal input for two LIF neurons","title":"Sinuisodal input for two LIF neurons","text":"This page was generated using Literate.jl.","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"EditURL = \"https://github.com/dylanfesta/SpikingRNNs.jl/blob/master/examples/lif_2neurons.jl\"","category":"page"},{"location":"lif_2neurons/#Two-LIF-neurons","page":"Two LIF neurons","title":"Two LIF neurons","text":"","category":"section"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"In this example I show two LIF neuron, one excitatory and one inhibitory, connected together. I plot the voltage traces, the internal currents, the refractoriness.","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"I access to the internal variables directly, but in the next examples I will be using recorder objects.","category":"page"},{"location":"lif_2neurons/#Initialization","page":"Two LIF neurons","title":"Initialization","text":"","category":"section"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"using LinearAlgebra,Statistics,StatsBase\nusing Plots,NamedColors ; theme(:default)\nusing SparseArrays\nusing SpikingRNNs; const global S = SpikingRNNs\n\nfunction onesparsemat(w::Real)\n  return sparse(cat(w;dims=2))\nend;\nnothing #hide","category":"page"},{"location":"lif_2neurons/#Parameters","page":"Two LIF neurons","title":"Parameters","text":"","category":"section"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"const dt = 1E-3","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"two LIF neurons, E and I","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"const τe = 0.2 # time constant for dynamics\nconst τi = 0.1\nconst cap_e = τe # capacitance\nconst cap_i = τi\nconst vth = 10.  # action-potential threshold\nconst vreset = -5.0 # reset potential\nconst vleak = -5.0 # leak potential\nconst τrefre = 0.2 # refractoriness\nconst τrefri = 0.3\nconst τpcd = 0.2 # synaptic kernel decay\n\nconst ps_e = S.PSIFNeuron(1,τe,cap_e,vth,vreset,vleak,τrefre)\nconst ps_i = S.PSIFNeuron(1,τi,cap_i,vth,vreset,vleak,τrefri);\nnothing #hide","category":"page"},{"location":"lif_2neurons/#Define-static-inputs","page":"Two LIF neurons","title":"Define static inputs","text":"","category":"section"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"const h_in_e = 10.1 - vleak\nconst h_in_i = 0.0\nconst in_e = S.IFInputCurrentConstant([h_in_e,])\nconst in_i = S.IFInputCurrentConstant([h_in_i,])","category":"page"},{"location":"lif_2neurons/#Define-connections","page":"Two LIF neurons","title":"Define connections","text":"","category":"section"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"connect E <-> I , both ways, but no autapses","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"const conn_in_e = S.ConnectionIFInput([1.,])\nconst conn_in_i = S.ConnectionIFInput([1.,])\nconst w_ie = 30.0\nconst w_ei = 40.0\nconst conn_ie = S.ConnectionIF(τpcd,onesparsemat(w_ie))\nconst conn_ei = S.ConnectionIF(τpcd,onesparsemat(w_ei);is_excitatory=false);\nnothing #hide","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"connected populations","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"const pop_e = S.Population(ps_e,(conn_ei,ps_i),(conn_in_e,in_e))\nconst pop_i = S.Population(ps_i,(conn_ie,ps_e),(conn_in_i,in_i));\nnothing #hide","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"that's it, let's make the network","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"const network = S.RecurrentNetwork(dt,(pop_e,pop_i));\n\n# # src","category":"page"},{"location":"lif_2neurons/#Network-simulation","page":"Two LIF neurons","title":"Network simulation","text":"","category":"section"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"const Ttot = 15.\nconst times = (0:network.dt:Ttot)\nnt = length(times)","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"set initial conditions","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"ps_e.state_now[1] = vreset\nps_i.state_now[1] = vreset + 0.95*(vth-vreset)","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"things to save","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"myvse = Vector{Float64}(undef,nt) # voltage\nmyfiringe = BitVector(undef,nt) # spike raster\nmyrefre = similar(myfiringe)  # if it is refractory\neicurr = similar(myvse)  # e-i current\n\nmyvsi = Vector{Float64}(undef,nt)\nmyfiringi = BitVector(undef,nt)\nmyrefri = similar(myfiringe)\niecurr = similar(myvsi)\n\nfor (k,t) in enumerate(times)\n  S.dynamics_step!(t,network)\n  myvse[k] = ps_e.state_now[1]\n  myfiringe[k]=ps_e.isfiring[1]\n  myrefre[k]=ps_e.isrefractory[1]\n  myvsi[k] = ps_i.state_now[1]\n  myfiringi[k]=ps_i.isfiring[1]\n  myrefri[k]=ps_i.isrefractory[1]\n  eicurr[k]=conn_ei.synaptic_kernel.trace[1]\n  iecurr[k]=conn_ie.synaptic_kernel.trace[1]\nend;\nnothing #hide","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"add spikes for plotting purposes, the eight is set arbitrarily to three times the firing threshold","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"myvse[myfiringe] .= 3 * vth\nmyvsi[myfiringi] .= 3 * vth\n\ntheplot = let  plt=plot(times,myvse;leg=false,linewidth=1,\n  ylabel=\"E (mV)\",\n  color=colorant\"Midnight Blue\") # the green line indicates when the neuron is refractory\n  plot!(plt,times, 20.0*myrefre; opacity=0.6, color=:green,linewidth=1)\n  plti=plot(times,myvsi;leg=false,linewidth=1,\n     ylabel=\"I (mV)\",\n    color=colorant\"Brick Red\")\n  plot!(plti,times, 20.0*myrefri; opacity=0.6, color=:green)\n  pltcurr=plot(times, [ iecurr eicurr]; leg=false,\n    linewidth=1, ylabel=\"E/I connection curr.\",\n    color=[colorant\"Teal\" colorant\"Orange\"])\n  plot(plt,plti,pltcurr; layout=(3,1),\n    xlabel=[\"\" \"\" \"time (s)\"])\nend;\nplot(theplot)","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"","category":"page"},{"location":"lif_2neurons/","page":"Two LIF neurons","title":"Two LIF neurons","text":"This page was generated using Literate.jl.","category":"page"}]
}
