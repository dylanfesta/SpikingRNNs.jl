#=
Here I show the exact correspondence
my spiking implementation and code by
Litwin-Kumar / Auguste Schulz

Here I just compare weight matrix,definition of assemblies, input function

=#


# include inbuilt modules
using Distributions
using SparseArrays
using Dates
using LinearAlgebra
using Random
using Plots ; theme(:dark)
using Random ; Random.seed!(0)
using ProgressMeter
using SpikingRNNs; const global S = SpikingRNNs
using SparseArrays

# --------------------- include functions  ------------------------------------------------
# include runs the respective julia code, i.e. defined functions are then in the workspace

#function gennextsequencepretraining(stim::Array{Float64,2}, firstimg, novelimg; Nimg = 4, Ntrain = 20, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
function genstimparadigmpretraining(stim::Array{Float64,2}; Nimg = 4, Nass = 20, Ntrain = 20, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	# generate a stimulus sequence
	# where each assembly is
	# Nimg Number of images per sequence
	# Nreps Number of seqeunce repititions in one block
	# Nseq Number of sequences
	# Nblocks Number of block repititions

	if Ntrain != 0
		Nstim = Nass * Ntrain
		tempstim = zeros(Nstim, 4)

		#tempstim[1,1] = firstimg
		# repeat sequence several times, i.e. assemblie numbers 123412341234...
		#tempstim[:,1] = repeat(firstimg:(firstimg+Nimg-1), outer = Nreps)
		# ensure that images are shuffled but never the same image after the other
		images = 1:Nass
		assemblysequence = shuffle(copy(images))
		for rep = 2:Ntrain
			shuffleimg = shuffle(images)
			#println("begin $(shuffleimg[1])")
			while shuffleimg[1] == assemblysequence[end]
				println("shuffle again")
				shuffleimg = shuffle(images)
				#println("new begin $(shuffleimg[1])")
			end
		assemblysequence = vcat(assemblysequence, shuffleimg)
		end
		tempstim[:,1] .= assemblysequence
		if stim[end,3] == 0
			tempstim[1,2] = stimstart
		else
			tempstim[1,2] = stim[end,3]+lenpause
		end
		tempstim[1,3] = tempstim[1,2] + lenstim
		tempstim[1,4] = strength

		for i = 2:Nstim#Nimg*Nreps #number in image times
			# # lenstim ms stimulation and lenpause ms wait time
			tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
			tempstim[i,3] = tempstim[i,2]+lenstim
			tempstim[i,4] = strength
		end
		if stim[end,3] == 0
			stim = tempstim
		else
			stim = vcat(stim, tempstim)
		end
		#tempstim = nothing
	end
	return stim
end # function genstim


function gennextsequenceNonovel(stim::Array{Float64,2}, firstimg, novelimg; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	# generate a stimulus sequence
	# where each assembly is
	# Nimg Number of images per sequence
	# Nreps Number of seqeunce repititions in one block
	# Nseq Number of sequences
	# Nblocks Number of block repititions
	Nstim = Nimg*Nreps
	tempstim = zeros(Nstim, 4)
    #tempstim[1,1] = firstimg
	tempstim[:,1] = repeat(firstimg:(firstimg+Nimg-1), outer = Nreps)

	if stim[end,3] == 0
		tempstim[1,2] = stimstart
	else
    	tempstim[1,2] = stim[end,3]+lenpause
	end
    tempstim[1,3] = tempstim[1,2] + lenstim
    tempstim[1,4] = strength

    for i = 2:Nimg*Nreps #number in image times
        # lenstim ms stimulation and lenpause ms wait time
        tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
        tempstim[i,3] = tempstim[i,2]+lenstim
        tempstim[i,4] = strength
    end
	#tempstim[end-Nimg,1] = novelimg
	if stim[end,3] == 0
		stim = tempstim
	else
		stim = vcat(stim, tempstim)
	end
	#tempstim = nothing
    return stim
end # function genstim


function genstimparadigmNonovel(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)

	firstimages = collect(1:Nimg:Nimg*Nseq)
	novelimg = firstimages .+ Nimg*Nseq
	novelrep = collect(Nimg*Nseq:Nimg*Nseq+Nimg).+1

	# for img in firstimages
	# 	stimulus = gennextsequence!(stimulus, img)
	# end
	blockonset = []
	storefirstimages = copy(firstimages)
	for b = 1:Nblocks
		if b == 1
			for img in firstimages
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequenceNonovel(stimulus, img, img + Nseq * Nimg, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		else
			shuffleimg = shuffle(firstimages)
			while shuffleimg[1] == storefirstimages[end]
				shuffleimg = shuffle(firstimages)
			end
			storefirstimages = vcat(storefirstimages, shuffleimg)
			for img in shuffleimg # since we shuffle ensure that it is still the correct novel image
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequenceNonovel(stimulus, img, img + Nseq * Nimg,Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		end
	end
	# ensure mapping from novel img to right value despite shuffling
	for i = 1:length(novelimg)
		stimulus[stimulus[:,1] .== novelimg[i],1] .= novelrep[i]
	end

	# ensure that blockonset 1 when we start with an emptz array
	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, convert(Array{Int64,1}, blockonset)

end


function genstimparadigmnovelcont(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
  """	generate the stimulation paradigm ensure that for each block there is a different novel image and
  that two consecutive blocks are never the same
  
  generate seq. violation stimulation paradigm
  ABCABCABC....ABN1ABC
  DEFDEFDEF....DEN2DEF
  XYZXYZXYZ...XYN3XYZ
  DEFDEFDEF....DEN4DEF
  ABCABCABC....ABN1ABC
  """
  
    firstimages = collect(1:Nimg:Nimg*Nseq)
    novelimg = Nimg*Nseq + 1 # start counting from the first assembly after the core assemblies
    novelrep = collect(Nimg*Nseq:Nimg*Nseq+Nimg).+1
  
  
    storefirstimages = copy(firstimages)
    blockonset = []
  
    for b = 1:Nblocks
      #println("block $(b) -------------------")
      if b == 1
        for img in firstimages
          append!(blockonset, size(stimulus,1) + 1)
          stimulus = gennextsequencenovel(stimulus, img, novelimg, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
          novelimg += 1
        end
      else
        # ensure that the first images of two consecutive blocks differ
        shuffleimg = shuffle(firstimages)
        #println("begin $(shuffleimg[1])")
        while shuffleimg[1] == storefirstimages[end] && Nseq > 1 # added Nseq larger 1 cause otgerwise always the same
          println("shuffle again")
          shuffleimg = shuffle(firstimages)
          #println("new begin $(shuffleimg[1])")
        end
        storefirstimages = vcat(storefirstimages, shuffleimg)
        #println(storefirstimages)
        for img in shuffleimg  # since we shuffle ensure that it is still the correct novel image
          append!(blockonset, size(stimulus,1) + 1)
          stimulus = gennextsequencenovel(stimulus, img, novelimg ,Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
          novelimg += 1
        end
      end
    end
  
    # ensure that blockonset 1 when we start with an emptz array
    if blockonset[1] == 2
      blockonset[1] = 1
    end
    return stimulus, convert(Array{Int64,1}, blockonset)
  
  end
  
 
function gennextsequencenovel(stim::Array{Float64,2}, firstimg, novelimg; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	"""generate a stimulus sequence with novel stim
	Nimg Number of images per sequence
	Nreps Number of seqeunce repititions in one block
	Nseq Number of sequences
	Nblocks Number of block repititions"""
	Nstim = Nimg*Nreps
	tempstim = zeros(Nstim, 4)
    #tempstim[1,1] = firstimg
	# repeat sequence several times, i.e. assemblie numbers 123412341234...
	tempstim[:,1] = repeat(firstimg:(firstimg+Nimg-1), outer = Nreps)
	if stim[end,3] == 0
		tempstim[1,2] = stimstart
	else
    	tempstim[1,2] = stim[end,3]+lenpause
	end
    tempstim[1,3] = tempstim[1,2] + lenstim
    tempstim[1,4] = strength

    for i = 2:Nimg*Nreps #number in image times
        # # lenstim ms stimulation and lenpause ms wait time
        tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
        tempstim[i,3] = tempstim[i,2]+lenstim
        tempstim[i,4] = strength
    end
	tempstim[end-Nimg,1] = novelimg
	if stim[end,3] == 0
		stim = tempstim
	else
		stim = vcat(stim, tempstim)
	end
	#tempstim = nothing
    return stim
end # function genstim

 
function genstimparadigmssa(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
    """
    generate SSA stimulation paradigm
    A A A A A A A B A A A ... Block 1
    B B B B B 				 Block 2 ...
    Nreps 20 times
    Blocks
    """
    Nimg = 2
    Nreps = 5
    Nseq = 5
    Nblocks = 10
    stimulus, blockonset = genstimparadigmnovelcont(stimulus, Nimg = 2, Nreps = 5, Nseq = 5, Nblocks = 10, stimstart = stimstart, lenstim = 100, lenpause = 300, strength = strength )
  
    #last_onset = blockonset[1]
    freq = 1
    dev = 2
    for bb =1:length(blockonset)
      #idxfreq = collect(blockonset[bb]:(blockonset[bb]+Nimg*Nreps))
      freq_start = blockonset[bb]
      freq_end = blockonset[bb]+Nimg*Nreps
      dev_end = blockonset[bb]+Nimg*Nreps
      dev_start = blockonset[bb]+Nreps
      if bb == length(blockonset)
        stimulus[freq_start:end,1] .= freq
        #println(idxfreq)
        #idxdev = collect(blockonset[bb]+Nreps:Nreps:(blockonset[bb]+Nimg*Nreps))
        stimulus[dev_start:Nreps:end,1] .= dev
        #printl
      else
        stimulus[freq_start:freq_end,1] .= freq
        #println(idxfreq)
        #idxdev = collect(blockonset[bb]+Nreps:Nreps:(blockonset[bb]+Nimg*Nreps))
        stimulus[dev_start:Nreps:dev_end,1] .= dev
        #println(idxdev)
        # for the nect block switch deviant and frequent
      end
      temp =freq
      freq = dev
      dev = temp
    end
  
    return stimulus, blockonset
end
  



function initweights((Ne, Ni, p, Jee0, Jei0, Jie, Jii))
	""" initialise the connectivity weight matrix based on initial
	E-to-E Jee0 initial weight e to e plastic
	I-to-E Jei0 initial weight i to e plastic
	E-to-I Jie0 constant weight e to i not plastic
	I-to-I Jii constant weight i to i not plastic"""
	Ncells = Ne+Ni

	# initialise weight matrix
	# w[i,j] is the weight from pre- i to postsynaptic j neuron
	weights = zeros(Float64,Ncells,Ncells)
	weights[1:Ne,1:Ne] .= Jee0
	weights[1:Ne,(1+Ne):Ncells] .= Jie
	weights[(1+Ne):Ncells,1:Ne] .= Jei0
	weights[(1+Ne):Ncells,(1+Ne):Ncells] .= Jii
	# set diagonal elements to 0
	# for cc = 1:Ncells
	# 	weights[cc,cc] = 0
	# end
	weights[diagind(weights)] .= 0.0
	# ensure that the connection probability is only p
	weights = weights.*(rand(Ncells,Ncells) .< p)
	return weights
end

function weightpars(;Ne = 4000, Ni = 1000, p = 0.2 )
	"""Ne, Ni number of excitatory, inhibitory neurons
	p initial connection probability"""
	Jee0 = 2.86 #initial weight e to e plastic
	Jei0 = 48.7 #initial weight i to e plastic
	Jie = 1.27 #constant weight e to i not plastic
	Jii = 16.2 #constant weight i to i not plastic
	return Ne,Ni,p,Jee0,Jei0,Jie,Jii
end

function initassemblymembers(;Nassemblies = 20, pmember = .05, Nmembersmax = 300, Ne = 4000)
	"""Nassemblies number of assemblies
	pmember probability of belonging to any assembly
	Nmembersmax maximum number of neurons in a population (to set size of matrix)

	set up excitatory assemblies"""
	#seed = 1
	#Random.seed!(seed) # ensure same Assemblies when repeating
	assemblymembers = ones(Int,Nassemblies,Nmembersmax)*(-1)
	for pop = 1:Nassemblies
		members = findall(rand(Ne) .< pmember)
		assemblymembers[pop,1:length(members)] = members
	end
	#println(assemblymembers)
	return assemblymembers
end

function initinhibassemblymembers(;Nassemblies = 20, pmember = .15, Nmembersmax = 200, Ne = 4000, Ni = 1000)
	"""Nassemblies number of assemblies
	pmember probability of belonging to any assembly
	Nmembersmax maximum number of neurons in a population (to set size of matrix)

	set up inhibitory assemblies
	higher connection probability"""
	#seed = 1
	#Random.seed!(seed) # ensure same Assemblies when repeating
	inhibmembers = ones(Int,Nassemblies,Nmembersmax)*(-1)
	for pop = 1:Nassemblies
		members = findall(rand(Ni) .< pmember) .+ Ne
		inhibmembers[pop,1:length(members)] = members
	end
	#println(assemblymembers)
	return inhibmembers
end


#include("../simulation/runsimulation_inhibtuning.jl")
#include("../simulation/sequencefunctions.jl")
#include("../simulation/helperfunctions.jl")
#include("../evaluation/evaluationfunctions.jl")

# --------------------- initialise simulation --------------------------------------------

# Define number of excitatory and inhibitory neurons
const Ne = 4000
const Ni = 1000

Ncells = Ne + Ni


weights_au = initweights(weightpars(Ne = Ne, Ni = Ni))
weights_compare = permutedims(weights_au)
heatmap(weights_au)

pweights=0.2
Jee = 2.86 #initial weight e to e plastic
weights_ee = S.make_sparse_weights(Ne,Ne,pweights,Jee)
Jei = 48.7 #initial weight i to e plastic
weights_ei = S.make_sparse_weights(Ne,Ni,pweights,Jei)
Jie = 1.27 #constant weight e to i not plastic
weights_ie = S.make_sparse_weights(Ni,Ne,pweights,Jie)
Jii = 16.2 #constant weight i to i not plastic
weights_ii = S.make_sparse_weights(Ni,Ni,pweights,Jii)

weights_all = Matrix(vcat(hcat(weights_ee,weights_ei), hcat(weights_ie,weights_ii)))

heatmap(weights_all)
heatmap(weights_compare)


##
# Nimg 3, Nreps 20, Nseq 10, Nblocks 1, lenstim 300, strength 12, Ntrain 5, adjustfactor 1.0, adaptive neurons false, inhibfactor 0.1
ARGS = ["3", "20", "10", "1","300", "12", "5", "10", "0", "10"] # unique sequences
# ARGS = ["3", "20", "5", "10","300", "12", "5", "10", "0", "10"] # repeated sequences

# stimulus parameters
Nimg = parse(Int64, ARGS[1]) # number of stimuli per sequence
Nreps = parse(Int64, ARGS[2]) # number of repetitions per sequence cycle
Nseq = parse(Int64, ARGS[3]) # number of sequences
Nblocks = parse(Int64, ARGS[4]) # number of sequence block repetitions
stimstart = 1_000  # 4000 # start time of stimulation in ms
lenstimpre = parse(Int64, ARGS[5]) # duration of assembly stimulation per image in ms
lenpausepre = 0 # duration of no stimulation between two images in ms
strengthpre = parse(Int64, ARGS[6]) # strength of the stimulation in kHz added to baseline input
lenstim = lenstimpre # duration of assembly stimulation per image in ms
lenpause = lenpausepre # duration of no stimulation between two images in ms
strength = strengthpre # strength of the stimulation in kHz added to baseline input
Ntrain = parse(Int64, ARGS[7]) # number of pretraining iterations
Nass = (Nimg) * Nseq +  Nseq * Nblocks # total number of assemblies Nimg * Nseq + all novelty assemblies
##

p_ase = 0.05
p_asi = 0.15
Nassemblies = 40

assemblymembers = initassemblymembers(Nassemblies = Nass,Ne = Ne)
# inhibitory tuning
inhibassemblies = initinhibassemblymembers(Nassemblies = Nass, Ne = Ne, Ni = Ni)

assemblies_e = map(1:Nassemblies) do _
  sort(findall(rand(Ne) .< p_ase))
end
assemblies_i = map(1:Nassemblies) do _
  sort(findall(rand(Ni) .< p_asi))
end

map(r->count(r.>0),eachrow(assemblymembers))
length.(assemblies_e)


map(r->count(r.>0),eachrow(inhibassemblies))
length.(assemblies_i)

##


stimulus = zeros(1, 4)	# initialisation of the stimulus


# ---------------------- generate the stimulus --------------------------------


# specify aspects of the stimulation paradigm
withnovelty 	= true 	# novelty response
pretrainig 		= true	# include pretraining phase

# ----------- standard setting when the following three false -----------------
shuffledstim 	= false	# shuffle the stimulus randomly
lastshuffled 	= false	# shuffle the last two stimuli
reducednovelty 	= false	# reduce the stimulation strength of the novel stimulus


if withnovelty
	if pretrainig
		# --------- main stim generation function ---------------
		stimulus = genstimparadigmpretraining(stimulus, Nass = Nass, Ntrain = Ntrain, stimstart = stimstart, lenstim = lenstimpre, lenpause = lenpausepre, strength = strengthpre)
		lenpretrain = size(stimulus,1)
	end

	if shuffledstim
		if pretrainig
			stimulus, blockonset = genstimparadigmnovelshuffledpretrain(stimulus, lenpretrain, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			tag = "repeatedsequencesshuffled.h5"

		else
			stimulus, blockonset = genstimparadigmnovelshuffled(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
			tag = "repeatedsequencesshuffled.h5"
		end
	else # if shuffled stim

		# --------- main stim generation function ---------------
		stimulus, blockonset = genstimparadigmnovelcont(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
		tag = "repeatedsequences.h5"


	end
else # if with novelty
	# even if no novelty assemblies are shown still train with all novelty assemblies
	if pretrainig
		#println(stimulus)
		stimulus = genstimparadigmpretraining(stimulus, Nass = Nass, Ntrain = Ntrain, stimstart = stimstart, lenstim = lenstimpre, lenpause = lenpausepre, strength = strengthpre)
		#println(stimulus)
		lenpretrain = size(stimulus,1)
		tag = "repeatedsequences_imprinting.h5"

	end
	tag = "repeatedsequencesnoNovelty.h5"
	stimulus, blockonset = genstimparadigmNonovel(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
end



function inputfun(t::R,i::Integer,low::R,
    stimulus_mat::Matrix{Float64},
    assembly_v::Vector{Vector{Int64}};scale_factor::Float64=1.0) where R<:Real
  # catch the first or last
  (!(stimulus_mat[1,2] <= t <= stimulus_mat[end,3])) && (return low)
  stimstarts = view(stimulus_mat,:,2)
  # find row number
  therow = searchsortedfirst(stimstarts,t) - 1
  # account for stim duration
  (t >= stimulus_mat[therow,3]) && (return low)
  # ok, the assembly is active!
  as_v = assembly_v[Int64(stimulus_mat[therow,1])]
  if _is_in_assembly(i,as_v)
    return low + scale_factor*stimulus_mat[therow,4] # add to stimulus 
  else
    return low
  end
  return low # in case the compiler is dumb
end




function inputfun_upper(t::R,i::Integer,low::R,stimmat::Matrix{R},asv::Vector;
   scale_factor::Float64=1.0) where R<:Real
  return max(inputfun(t,i,low,stimmat,asv;scale_factor=scale_factor), 
              inputfun(t+2dt,i,low,stimmat,asv;scale_factor=scale_factor))
end


function _is_in_assembly(neu_idx::Integer,as_vec::Vector{<:Integer})
  if as_vec[1] <= neu_idx <= as_vec[end]
    idx = searchsortedfirst(as_vec,neu_idx)
    return as_vec[idx] == neu_idx
  else
    return false
  end
end
const assembly_vectors = 
    map(eachrow(hcat(assemblymembers,inhibassemblies))) do r
    sort(filter(>(0),r))
end
as_ordering = let Ntot=Ncells
  S.order_by_pattern_idxs(assembly_vectors,Ntot)
end

const stimulus_sec = copy(stimulus)
stimulus_sec[:,[2,3]] ./= 1E3
stimulus_sec[:,4] .*= 1E3

_ = let low_e = 2E3, low_i = 3E3,
  _scal = 0.1
  ratefun = function (t,i) 
    if i<=Ne 
      inputfun(t,i,low_e,stimulus_sec,assembly_vectors)
    else
      inputfun(t,i,low_i,stimulus_sec,assembly_vectors,scale_factor=_scal)
    end
  end
  times = range(0,100;length=200)
  nshow = Ncells
  neushow = as_ordering[1:nshow]
  vals = hcat([ ratefun.(t,neushow) for t in times]...)
  # vals_toplot = vals[as_ordering,:][1:nshow,:]
  heatmap(times,1:nshow,vals)
end

# now mine!

##
Ttot = 60.0
lowrate_e =2E3 
highrate_e =12E3
lowrate_i =3E3 
highrate_i = 12E3
scale_fact_i = 0.1
Δt_as = 0.3
t_as_delay = 1.0
Δt_blank = 0.0

inputfun_e = S.PatternPresentation(Δt_as,Ttot,lowrate_e,lowrate_e+highrate_e,Ne,
    assemblies_e;t_pattern_delay=t_as_delay,Δt_pattern_blank=Δt_blank)
inputfun_i = S.PatternPresentation(Δt_as,Ttot,lowrate_i,lowrate_i+scale_fact_i*highrate_i,
  Ni, assemblies_i;t_pattern_delay=t_as_delay,Δt_pattern_blank=Δt_blank)
# force same sequence
inputfun_i.sequence .= inputfun_e.sequence


##

as_ordering_mine = let 
  assemblies_joint=map(ei->sort(vcat(ei[1],Ne.+ei[2])),zip(assemblies_e,assemblies_i))
  S.order_by_pattern_idxs(assemblies_joint,Ne+Ni)
end

##

_ = let 
  ratefun = function (t,i) 
    if i<=Ne 
      inputfun_e(t,i)
    else
      inputfun_i(t,i-Ne)
    end
  end
  times = range(0,100;length=200)
  nshow = Ncells
  neushow = as_ordering_mine[1:nshow]
  vals = hcat([ ratefun.(t,neushow) for t in times]...)
  # vals_toplot = vals[as_ordering,:][1:nshow,:]
  heatmap(times,1:nshow,vals)
end