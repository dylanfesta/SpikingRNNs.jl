#=
Here I show the exact correspondence
my spiking implementation and code by
Litwin-Kumar / Auguste Schulz
Compare rates without assembly input and without plasticity, first
=#


# include inbuilt modules
using Distributions
using Dates
using LinearAlgebra
using Random
using Distributed
using Plots ; theme(:dark)
using Random ; Random.seed!(0)

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
# Set integration timestep
dt 	= 0.1 #integration timestep in ms

ifiSTDP = false # true 		#  include inhibitory plasticity
ifwadapt = false	#  consider AdEx or Ex IF neurons

# --------------------- generate the stimulus --------------------------------------------

# In case this file is not run from the command line specify the ARGS list of strings
# Nimg 3, Nreps 20, Nseq 10, Nblocks 1, lenstim 300, strength 12, Ntrain 5, adjustfactor 1.0, adaptive neurons false, inhibfactor 0.1
ARGS = ["3", "20", "10", "1","300", "12", "5", "10", "0", "10"] # unique sequences
# ARGS = ["3", "20", "5", "10","300", "12", "5", "10", "0", "10"] # repeated sequences

# stimulus parameters
Nimg = parse(Int64, ARGS[1]) # number of stimuli per sequence
Nreps = parse(Int64, ARGS[2]) # number of repetitions per sequence cycle
Nseq = parse(Int64, ARGS[3]) # number of sequences
Nblocks = parse(Int64, ARGS[4]) # number of sequence block repetitions
stimstart = 10_000  # 4000 # start time of stimulation in ms
lenstimpre = parse(Int64, ARGS[5]) # duration of assembly stimulation per image in ms
lenpausepre = 0 # duration of no stimulation between two images in ms
strengthpre = parse(Int64, ARGS[6]) # strength of the stimulation in kHz added to baseline input
lenstim = lenstimpre # duration of assembly stimulation per image in ms
lenpause = lenpausepre # duration of no stimulation between two images in ms
strength = strengthpre # strength of the stimulation in kHz added to baseline input
Ntrain = parse(Int64, ARGS[7]) # number of pretraining iterations
Nass = (Nimg) * Nseq +  Nseq * Nblocks # total number of assemblies Nimg * Nseq + all novelty assemblies

# adjustment of learning rates
adjustfactor = parse(Int64, ARGS[8])/10
adjustfactorinhib = adjustfactor # ensure they are equal for inhibitory and exc. plasticity

# adaptive currents
if parse(Int64, ARGS[9]) == 1
	ifwadapt = true
end

# strength of the inhibitory tuning
inhibfactor = parse(Int64, ARGS[10])/100 # 100 more refined steering possibility to just switch it off


stimparams = [Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength] # store stimulus param array in hdf5 file
stimparams_prestim = [Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength, lenstimpre, lenpausepre, strengthpre, Ntrain] # store stimulus param array in hdf5 file

# initialise stimulus array
stimulus = zeros(1, 4)	# initialisation of the stimulus
# nohup julia initsim_sequence_violation_ARGS.jl 3 20 5 10 300 12 5 10 0 10 &> ../tmp/Pre5_Final_Run_Block_nonadapt_10_inhibtuning_nohetero_tunefactor_10_shortpretrain.txt &


# specify a tag which is added at the end of the filename to account for different stimulation paradigms
#tag = "repseq.h5"

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


# set length of pretraining to 0 if no pretraining
if Ntrain == 0
	lenpretrain = 0
end


# get sequence order
blockidx = collect(lenpretrain + 1:Nreps*Nimg:size(stimulus,1)) # get all stimulus indices when a new sequence starts
seqnumber = stimulus[blockidx,1]
for n = 2:Nseq
    seqnumber[seqnumber .== (1 + (n-1)*Nimg)] .= n
end



# Switch last two stimuli instead of novel stimulus
if lastshuffled
	if pretrainig

		idx = findall(stimulus[lenpretrain + 1:end,1] .> Nseq*Nimg)
		vals = stimulus[idx.+(lenpretrain),1]
		stimulus[idx.+(lenpretrain),1] .= stimulus[idx.+(lenpretrain-1),1]
		stimulus[idx.+(lenpretrain-1),1] .= stimulus[idx.+(lenpretrain-1),1] .+ 1
		tag = "pretrainshufflenovelty" * tag
	else
		idx = findall(stimulus[:,1] .> Nseq*Nimg)
		vals = stimulus[idx,1]
		stimulus[idx,1] = stimulus[idx.-1,1]
		stimulus[idx.-1,1] = stimulus[idx,1] .+ 1
		tag = "shufflenovelty" * tag

	end # if pretraining

end # if lastshuffled


# reduce novelty input by a certain factor a different one for each sequence
# to infer the relevance of the novelty stimulus strength
lastimages = collect(Nimg:Nimg:Nimg*Nseq)
reducefactor =  collect(0:(1/(Nseq-1)):1)
secondtolastimage = lastimages.-1

if reducednovelty
	if pretrainig

		idx = findall(stimulus[lenpretrain + 1:end,1] .> Nseq*Nimg)
		vals = stimulus[idx.+(lenpretrain),1]
		for i in idx
			for im = 1:length(secondtolastimage)
				if stimulus[i+(lenpretrain-1),1] == secondtolastimage[im]# if previous image is certain second to last image of sequence reduce by corresponding factor
					stimulus[i+(lenpretrain),4] = stimulus[i+(lenpretrain),4]*reducefactor[im]
				end
			end
		end
		tag = "reducednovelty$(reducefactor)" * tag
	else
		idx = findall(stimulus[:,1] .> Nseq*Nimg)
		vals = stimulus[idx,1]
		for i in idx
			for im = 1:length(secondtolastimage)
				if stimulus[i-1,1] == secondtolastimage[im]# if previous image is certain second to last image of sequence reduce by corresponding factor
					stimulus[i,4] = stimulus[i,4]*reducefactor[im]
				end
			end
		end
		tag = "reducednovelty$(reducefactor)" * tag
	end

end


# simulation run time
T = stimulus[end,3]+lenpause # last stimulation time + pause duration
println(T)

println("Simulation run time: $T")





# --------------------- initialise savefile ---------------------------------------------------------

# initialise savefile and avoid overwriting when identical parameters are used
datetime = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")

filesavename = "seq_violation_$(inhibfactor)_dur$(T)msNblocks$(Nblocks)Ntrain$(Ntrain)lenstim$(lenstim)lenpause$(lenpause)Nreps$(Nreps)strength$(strength)wadapt$(ifwadapt)iSTDP$(ifiSTDP)RateAdjust$(adjustfactor)Time"

savefile = "../data/"*filesavename * datetime * tag
println(savefile)

# --------------------- initialise weights and assemblymembers -----------------------------------------

weights = initweights(weightpars(Ne = Ne, Ni = Ni))
# excitatory tuning
assemblymembers = initassemblymembers(Nassemblies = Nass,Ne = Ne)
# inhibitory tuning
inhibassemblies = initinhibassemblymembers(Nassemblies = Nass, Ne = Ne, Ni = Ni)

winit = copy(weights) # make a copy of initial weights as weights are updated by simulation


# --------------------- store relevant initialisation parameters -------------------------------------------
#=
h5write(savefile, "initial/stimulus", stimulus)
h5write(savefile, "initial/lengthpretrain", lenpretrain)#
h5write(savefile, "initial/stimparams", stimparams)
h5write(savefile, "initial/stimparams_prestim", stimparams_prestim)
h5write(savefile, "initial/seqnumber", seqnumber)
h5write(savefile, "initial/idxblockonset", blockonset)
h5write(savefile, "initial/weights", weights)
h5write(savefile, "initial/assemblymembers", assemblymembers)
h5write(savefile, "initial/inhibassemblies", inhibassemblies)

h5write(savefile, "params/T", T)
h5write(savefile, "params/Ne", Ne)
h5write(savefile, "params/Ni", Ni)

=#


# Define storage decisions
Ndec = 1
storagedec = zeros(Bool,Ndec)
storagetimes = ones(Int,Ndec)*1000

# Stroage decisions
storagedec[1] = true  # store times when spikes occured

# --------------------- initialise spiketime matrix ID - t  -----------------------------------------

Tstore = 1000 # average duration of one spiketime matrix
avgrate = 10 # Hz
Nspikesmax = Ncells*Tstore*avgrate/1000 # Nr. neurons x run time in seconds x avg. firing rate
spiket = zeros(Int32,Int(Nspikesmax),2)


if Ntrain == 0 # switch back to 1 as it is used in run simulation julia indexing starts at 1
	lenpretrain = 1
end
# --------------------- run simulation ---------------------------------------------------------------
##

##############################################################################
#
# This code is part of the publication:
# https://www.biorxiv.org/content/10.1101/2020.11.30.403840v1
#
# The generation of cortical novelty responses through inhibitory plasticity
# Auguste Schulz*, Christoph Miehl*, Michael J. Berry II, Julijana Gjorgjieva
#
# * equal contribution
#
##############################################################################

# Additional information and instruction in README


function runsimulation_inhibtuning(ifiSTDP,ifwadapt,stimparams,stimulus::Array{Float64,2},
	weights::Array{Float64,2}, assemblymembers::Array{Int64,2},
	spiketimes::Array{Int32,2},storagedec::Array{Bool,1},
	storagetimes::Array{Int64,1}, savefile::String, lenpretrain,
	inhibassemblies::Array{Int64,2}; dt = 0.1, T = 2000,
	adjustfactor = 1, adjustfactorinhib=1, inhibtuing = true, inhibfactor = 0.1,
	bwfactor=100, tauw_adapt=150)

"""	Runs a new simulation of a plastic E-I spiking neural network model where both E and I
	neurons receive tuned input

	Inputs:
		ifiSTDP boolean: if inhibitory spike timing plasticity is active
		ifwadapt boolean: if an intrisic adaptive current is included for E neurons
		stimparams array: with stimulus parameters
		stimulus array: stimulus constisting of [stimulated assembly, start time, stop time, rate increase]
		weights array: initial weight matrix
		assemblymembers array: members of an excitatory assembly (group of E neurons tuned to the same stimulus)
		spiketimes array: spiketimes of all neuron [spike time, neuron id] (in julia speed is increased by prior initialisation)
		storagedec array of booleans: storage decisions what should be stored
		storagetimes array: how often each of the items should be stored
		savefile string: name of the file to store to
		lenpretrain int: duration of the pretraining phase prior to starting the real simulation in stimulus indices
		inhibassemblies array: members of an inhibitory "assembly" (group of I neurons tuned to the same stimulus)
		dt float: integration timestep (optional)
		T float: total duration of the run (optional)
		adjustfactor float: adjust factor of the E-to-E plasticity after pretraining to allow for switching off plasticity
		adjustfactorinhib float: adjust factor of the I-to-E plasticity after pretraining to allow for switching off plasticity
		inhibtuing boolean: if inhibitory neurons are tuned to stimuli as well
		inhibfactor float: the scaling factor by how much the inhibitory neurons are driven by an external stimulus compared to the excitatory tuned neurons
		bwfactor float: scaling factor of the adaptive currents (only required for adaptive experiments)
		tauw_adapt float: timescale of the adaptive currents (only required for adaptive experiments)

	Output:

		totalspikes array: set of last spikes
		several run parameters are stored in hdf5 files including all spiketimes of all neurons across the entire simulation
		"""

		# Naming convention
		# e corresponds to excitatory
		# i corresponds to inhibitory
		# x corresponds to external

		#membrane dynamics
		taue = 20 #e membrane time constant ms
		taui = 20 #i membrane time constant ms
		vreste = -70 #e resting potential mV
		vresti = -62 #i resting potential mV
		vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset mV
		eifslope = 2 #eif slope parameter mV
		C = 300 #capacitance pF
		erev = 0 #e synapse reversal potential mV
		irev = -75 #i synapse reversal potntial mV
		vth0 = -52 #initial spike voltage threshold mV
		thrchange = false # can be switched off to have vth constant at vth0
		ath = 10 #increase in threshold post spike mV
		tauth = 30 #threshold decay timescale ms
		vreset = -60 #reset potential mV
		taurefrac = 1 #absolute refractory period ms
		aw_adapt = 4 #adaptation parameter a nS conductance
		bw_adapt = bwfactor*0.805 #adaptation parameter b pA current
    #=
		if T > 1 # this should not be saved for the precompile run
			h5write(savefile, "params/MembraneDynamics", [taue,taui, vreste,vresti , vpeak , eifslope, C, erev, irev, vth0, thrchange, ath, tauth, vreset, taurefrac, aw_adapt,bw_adapt, tauw_adapt ])
			h5write(savefile, "params/STDPwadapt", [Int(ifiSTDP),Int(ifwadapt)])
			h5write(savefile, "params/bwfactor", bwfactor)
			h5write(savefile, "params/adjustfactor", adjustfactor)
			h5write(savefile, "params/adjustfactorinhib", adjustfactorinhib)

		end
    =#
		# total number of neurons
		Ncells = Ne+Ni

		# synaptic kernel
		tauerise = 1 #e synapse rise time
		tauedecay = 6 #e synapse decay time
		tauirise = .5 #i synapse rise time
		tauidecay = 2 #i synapse decay time

		# external input
		rex = 4.5 #external input rate to e (khz) since the timestep is one ms an input of 4.5 corresp to 4.5kHz
		rix = 2.5#2.25 #external input rate to i (khz) # on average this is reduced to 2.25
		println("Larger inhibitory input")
		# Ensure the overall inhibitory input remains the same when inhibiory tuning is included

		# ---------------- inhibitory tuning ------------------
		# reduce the overall inhibitory input by the amount added during each stimulus presentation
		memsinhib1 = inhibassemblies[1,inhibassemblies[1,:] .!= -1]
		# reduce the total inhibition based on the total added inhibition to stimulated inhibitory neurons
		added_inhib = length(memsinhib1)*stimulus[end,4]*inhibfactor
		println("initial rix ", rix)
		rix -= added_inhib/Ni
		println("reduced rix ", rix)


		# initial connectivity
		Jeemin = 1.78 #minimum ee weight  pF
		Jeemax = 21.4 #maximum ee weight pF

		Jeimin = 48.7 #minimum ei weight pF
		Jeimax = 243 #maximum ei weight pF


		Jex = 1.78 #external to e weight pF
		Jix = 1.27 #external to i weight pF
    #=
		if T > 1
			h5write(savefile, "params/Connectivity", [tauerise, tauedecay, tauirise, tauidecay, rex, rix, Jeemin, Jeemax, Jeimin, Jeimax, Jex, Jix])
		end
    =#
		#voltage based stdp (for alternative testing not used here)
		altd = .0008 #ltd strength pA/mV pairwise STDP LTD
		altp = .0014 #ltp strength pA/mV^2 triplet STDP LTP
		thetaltd = -70 #ltd voltage threshold mV
		thetaltp = -49 #ltp voltage threshold mV
		tauu = 10 #timescale for u variable ms
		tauv = 7 #timescale for v variable ms
		taux = 15 #timescale for x variable ms
    #=
		if T > 1
			h5write(savefile, "params/voltageSTDP", [altd, altp, thetaltd, thetaltp, tauu, tauv, taux])
		end
      =#

		#inhibitory stdp
		tauy = 20 #width of istdp curve ms
		eta = 1 #istdp learning rate pA
		r0 = .003 #target rate (khz)
    #=
		if T > 1
			h5write(savefile, "params/iSTDP", [tauy, eta, r0])
		end
    =#

		# triplet parameters
		tripletrule = false # true
		o1 = zeros(Float64,Ne);
		o2 = zeros(Float64,Ne);
		r1 = zeros(Float64,Ne);
		r2 = zeros(Float64,Ne);
    #=
		if T > 1
			h5write(savefile, "params/Triplet", Int(tripletrule))
		end
    =#
		tau_p = 16.8;        # in ms
		tau_m = 33.7;        # in s
		tau_x = 101.0;        # in s
		tau_y = 125.0;        # in s
		# init LTP and LTD variables
		A_2p = 7.5*10^(-10); # pairwise LTP disabled
		A_2m = 7.0*10^(-3);
		A_2m_eff = A_2m;       #effective A_2m, includes the sliding threshold
		A_3p = 9.3*10^(-3)
		A_3m = 2.3*10^(-4); # triplet LTP disabled
    #=
		if T > 1
			h5write(savefile, "params/TripletTausAs", [tau_p,tau_m, tau_x,tau_y , A_2p , A_2m, A_3p, A_3m])
		end
    =#
		# simulation parameters
		taurefrac = 1 #ms refractory preriod clamped for 1 ms
		dtnormalize = 20 #how often to normalize rows of ee weights ms heterosynaptic plasticity
		stdpdelay = 1000 #time before stdp is activated, to allow transients to die out ms
		dtsaveweights = 2000  # save weights  every 2000 ms
		# minimum and maximum of storing the weight matrices
		minwstore = 80
		modwstore = 10
    #=
		if T > 1
			h5write(savefile, "params/Normdt", dtnormalize)
			h5write(savefile, "params/dtsaveweights", dtsaveweights)
			h5write(savefile, "params/minwstore", minwstore)
			h5write(savefile, "params/modwstore", modwstore)
		end
    =#
		Nassemblies = size(assemblymembers,1) #number of assemblies
		Nmembersmax = size(assemblymembers,2) #maximum number of neurons in a population

		ttstimbegin = round(Integer,stimulus[lenpretrain,2]/dt) # is set to zero

		# stimulus parameters
		Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength = stimparams


		# Initialisation of spike arrays ------------------------------

		# spike count and spike times

		if storagedec[1]
			totalspikes = zeros(Int,Ncells)
		else
			spiketimes = nothing
			totalspikes = nothing
			totsp = nothing
		end
		totsp::Int64 = 0; # total numberof spikes
		spmax::Int64 = size(spiketimes,1); # maximum recorded spikes

		# further arrays for storing inputs and synaptic integration
		forwardInputsE = zeros(Float64,Ncells) #sum of all incoming weights from excitatpry inputs both external and EE and IE
		forwardInputsI = zeros(Float64,Ncells) #sum of all incoming weights from inhibitory inputs both II and EI
		forwardInputsEPrev = zeros(Float64,Ncells) #as above, for previous timestep
		forwardInputsIPrev = zeros(Float64,Ncells)

		xerise = zeros(Float64,Ncells) #auxiliary variables for E/I currents (difference of exponentials)
		xedecay = zeros(Float64,Ncells)
		xirise = zeros(Float64,Ncells)
		xidecay = zeros(Float64,Ncells)

		expdist = Exponential()

		v = zeros(Float64,Ncells) #membrane voltage
		nextx = zeros(Float64,Ncells) #time of next external excitatory input
		sumwee0 = zeros(Float64,Ne) #initial summed e weight, for normalization
		Nee = zeros(Int,Ne) #number of e->e inputs, for normalization
		rx = zeros(Float64,Ncells) #rate of external input


		# initialisation of membrane potentials and poisson inputs
		for cc = 1:Ncells
			v[cc] = vreset + (vth0-vreset)*rand()
			if cc <= Ne # excitatory neurons
				rx[cc] = rex
				nextx[cc] = rand(expdist)/rx[cc]
				for dd = 1:Ne
					sumwee0[cc] += weights[dd,cc]
					if weights[dd,cc] > 0
						Nee[cc] += 1
					end
				end
			else # inhibtory neurons
				rx[cc] = rix
				nextx[cc] = rand(expdist)/rx[cc]
			end
		end


		vth = vth0*ones(Float64,Ncells) #adaptive threshold
		wadapt = aw_adapt*(vreset-vreste)*ones(Float64,Ne) #adaptation current
		lastSpike = -100*ones(Ncells) #last time the neuron spiked
		trace_istdp = zeros(Float64,Ncells) #low-pass filtered spike train for istdp
		u_vstdp = vreset*zeros(Float64,Ne) # for voltage rule (not used here)
		v_vstdp = vreset*zeros(Float64,Ne) # for voltage rule (not used here)
		x_vstdp = zeros(Float64,Ne) # for voltage rule (not used here)


		# ---------------------------------- set up storing avg weights -----------------

		idxnovelty = zeros(Int32, 200)

		Nass = Nassemblies # number of distinct stimuli
		nm = zeros(Int32,Nass) # Number of memebers in this assembly
		inhibnm = zeros(Int32,Nass) # Number of memebers in this inhibitory "assembly"
		for mm = 1:Nass # loop over all assemblies
			#check when first -1 comes to determine number of neurons
			nm[mm] = sum(assemblymembers[mm,:].!=-1)
			inhibnm[mm] = sum(inhibassemblies[mm,:].!=-1)
		end

		# initialise avergae cross and inhbitiory assembly weights
		# Calculate them here instead of storing whole matrices (too large files)
		avgXassembly = zeros(Float64,Nass,Nass) # average cross assembly weights
		avgInhibXassembly = zeros(Float64,Nass,Nass) # average inhibitory to excitatory cross assembly weights
		avgItoassembly = zeros(Float64,Nass) # avergate I to E assembly weights
		avgnonmemstoassembly = zeros(Float64,Nass) # avergate weights from neurons not stimulus driven to exc. assemblies
		avgassemblytononmems = zeros(Float64,Nass) # avergate weights from exc. assemblies to neurons not stimulus driven
		avgassemblytonovelty = zeros(Float64,Nass) # avergate weights from exc. assemblies to novelty assemblies
		avgnoveltytoassembly = zeros(Float64,Nass) # avergate weights from exc. novelty assemblies to exc. assemblies

		# determine neurons not part of any excitatory assembly
		nonmems = collect(1:Ne)
		members = sort(unique(assemblymembers[assemblymembers .> 0]))
		deleteat!(nonmems, members)

		# total number of simulation steps, steps when to normalise, and save
		Nsteps = round(Int,T/dt)
		inormalize = round(Int,dtnormalize/dt)
		saveweights = round(Int,dtsaveweights/dt)

		# true time
		t::Float64 = 0.0
		tprev::Float64 = 0.0

		# counters how often a variable was saved
		weightstore::Integer = 0
		spikestore::Integer = 0
		novelstore::Integer = 0

		# bool counter if a neuron has just had a spike
		spiked = zeros(Bool,Ncells)


	#   ------------------------------------------------------------------------
	#
	#   				begin actual simulation
	#
	#   ------------------------------------------------------------------------

	# evaluate the run via @time
	@time	for tt = 1:Nsteps

				if mod(tt,Nsteps/100) == 1  #print percent complete
					print("\r",round(Int,100*tt/Nsteps))
				end

				forwardInputsE[:] .= 0.
				forwardInputsI[:] .= 0.
				t = dt*tt
				tprev = dt*(tt-1)

				# iterate over all stimuli in the passed stimulus array
				for ss = 1:size(stimulus)[1]

					if (tprev<stimulus[ss,2]) && (t>=stimulus[ss,2])  #just entered stimulation period
						ass = round(Int,stimulus[ss,1]) # TODO: refactor somewhat unfortunate naming of assembly

						# if assembly is a novelty assembly
						if ass > Nimg*Nseq && (t>stimulus[lenpretrain,2])
							# name idxnovelty still stems from time before defining novelty assembly directly just novelty assembly
							idxnovelty = assemblymembers[ass,assemblymembers[ass,:] .!= -1]
							novelstore += 1
							println("Ass :$(ass) time $(t) > $(stimulus[lenpretrain,2])")
							println("novelstore :$(novelstore)")

							# h5write(savefile, "novelty/indices$(novelstore)", assemblymembers[ass,assemblymembers[ass,:] .!= -1])
							#h5write(savefile, "novelty/assembly$(novelstore)", ass)
						end

						# stimulus tuning: increase the external stimulus to rx of the corresponding E assembly memebers
						mems = assemblymembers[ass,assemblymembers[ass,:] .!= -1]
						rx[mems] .+= stimulus[ss,4]

						# ---------------------- inhibtuning --------------------
						if inhibtuing
							memsinhib = inhibassemblies[ass,inhibassemblies[ass,:] .!= -1]
							rx[memsinhib] .+= stimulus[ss,4]*inhibfactor
						end

					end # if just entered stim period



					if (tprev<stimulus[ss,3]) && (t>=stimulus[ss,3]) #just exited stimulation period
						ass = round(Int,stimulus[ss,1])

						mems = assemblymembers[ass,assemblymembers[ass,:] .!= -1]
						rx[mems] .-= stimulus[ss,4]


						if inhibtuing
							memsinhib = inhibassemblies[ass,inhibassemblies[ass,:] .!= -1]
							rx[memsinhib] .-= stimulus[ss,4]*inhibfactor
						end

					end # if just left stim period

				end #end loop over stimuli

				if mod(tt,inormalize) == 0 #excitatory synaptic normalization
					for cc = 1:Ne
						sumwee = 0.
						for dd = 1:Ne
							sumwee += weights[dd,cc]
						end # dd

						for dd = 1:Ne
							if weights[dd,cc] > 0.
								weights[dd,cc] -= (sumwee-sumwee0[cc])/Nee[cc]
								if weights[dd,cc] < Jeemin
									weights[dd,cc] = Jeemin
								elseif weights[dd,cc] > Jeemax
									weights[dd,cc] = Jeemax
								end # if
							end # if
						end # dd for
					end # cc for
					#println("Normalised...")
				end #end normalization
        
        #=
				if mod(tt,saveweights) == 0 #&& (weightstore < 20 || mod(weightstore,100) == 0)#excitatory synaptic normalization
					weightstore += 1
					#if (weightstore < minwstore || mod(weightstore,modwstore) == 0)
						#h5write(savefile, "dursim/weights$(weightstore)_$(tt)", weights)
						for pre = 1:Nass # use maximum to be sure to capture it both for sequencelength and variablerepetitions
							# for inhib to assembly get avg. weights here pre is actually post
							avgnonmemstoassembly[pre] = getXassemblyweight(nonmems, assemblymembers[Int(pre),1:nm[Int(pre)]], weights)
							avgassemblytononmems[pre] = getXassemblyweight(assemblymembers[Int(pre),1:nm[Int(pre)]],nonmems, weights)
							if sum(idxnovelty) == 0 # before first novelty arose still zero
								avgnoveltytoassembly[pre] = 0
								avgassemblytonovelty[pre] = 0
							else
								avgnoveltytoassembly[pre] = getXassemblyweight(idxnovelty[idxnovelty.>0], assemblymembers[Int(pre),1:nm[Int(pre)]], weights)
								avgassemblytonovelty[pre] = getXassemblyweight(assemblymembers[Int(pre),1:nm[Int(pre)]],idxnovelty[idxnovelty.>0], weights)
							end
							avgItoassembly[pre] = getXassemblyweight(collect(Ne+1:Ncells), assemblymembers[Int(pre),1:nm[Int(pre)]], weights)
							for post = 1:Nass
								avgXassembly[pre,post] = getXassemblyweight(assemblymembers[Int(pre),1:nm[Int(pre)]], assemblymembers[Int(post),1:nm[Int(post)]], weights)
								# determine the average inhibitory assembly to excitatory assembly weight  define inhibnm
								avgInhibXassembly[pre,post] = getXassemblyweight(inhibassemblies[Int(pre),1:inhibnm[Int(pre)]], assemblymembers[Int(post),1:nm[Int(post)]], weights)

							end # post assemblies
						end # pre assemblies


            #=
						h5write(savefile, "dursimavg/avgXassembly$(weightstore)_$(tt)", avgXassembly)
						h5write(savefile, "dursimavg/avgInhibXassembly$(weightstore)_$(tt)", avgInhibXassembly)

						h5write(savefile, "dursimavg/avgItoassembly$(weightstore)_$(tt)", avgItoassembly)
						h5write(savefile, "dursimavg/avgnonmemstoassembly$(weightstore)_$(tt)", avgnonmemstoassembly)
						h5write(savefile, "dursimavg/avgassemblytononmems$(weightstore)_$(tt)", avgassemblytononmems)
						h5write(savefile, "dursimavg/avgassemblytonovelty$(weightstore)_$(tt)", avgassemblytonovelty)
						h5write(savefile, "dursimavg/avgnoveltytoassembly$(weightstore)_$(tt)", avgnoveltytoassembly)

						h5write(savefile, "dursimavg/Itoneuron1$(weightstore)_$(tt)", weights[Ne+1:Ncells,1][weights[Ne+1:Ncells,1].>0])# store only non zero inhibitory to neuron 1/2 weights
						h5write(savefile, "dursimavg/Itoneuron2$(weightstore)_$(tt)", weights[Ne+1:Ncells,2][weights[Ne+1:Ncells,2].>0]) # store only non zero inhibitory to neuron 1/2 weights
            =#

				end # mod(tt,saveweights) == 0
        =#
				fill!(spiked,zero(Bool)) # reset spike bool without new memory allocation

				for cc = 1:Ncells
					trace_istdp[cc] -= dt*trace_istdp[cc]/tauy

					while(t > nextx[cc]) #external input
						nextx[cc] += rand(expdist)/rx[cc]
						if cc <= Ne
							forwardInputsEPrev[cc] += Jex
						else
							forwardInputsEPrev[cc] += Jix
						end
					end

					xerise[cc] += -dt*xerise[cc]/tauerise + forwardInputsEPrev[cc]
					xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardInputsEPrev[cc]
					xirise[cc] += -dt*xirise[cc]/tauirise + forwardInputsIPrev[cc]
					xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardInputsIPrev[cc]

					if cc <= Ne # excitatory
						if thrchange
						vth[cc] += dt*(vth0 - vth[cc])/tauth;
						end
						wadapt[cc] += dt*(aw_adapt*(v[cc]-vreste) - wadapt[cc])/tauw_adapt;
						u_vstdp[cc] += dt*(v[cc] - u_vstdp[cc])/tauu;
						v_vstdp[cc] += dt*(v[cc] - v_vstdp[cc])/tauv;
						x_vstdp[cc] -= dt*x_vstdp[cc]/taux;


						# triplet accumulators
						r1[cc] += -dt*r1[cc]/tau_p # exponential decay of all
			            r2[cc] += -dt*r2[cc]/tau_x
						o1[cc] += -dt*o1[cc]/tau_m
			            o2[cc] += -dt*o2[cc]/tau_y
					end

					if t > (lastSpike[cc] + taurefrac) #not in refractory period
						# update membrane voltage

						ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise);
						gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise);

						if cc <= Ne #excitatory neuron (eif), has adaptation
							if ifwadapt
								dv = (vreste - v[cc] + eifslope*exp((v[cc]-vth[cc])/eifslope))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C - wadapt[cc]/C;
							else
								dv = (vreste - v[cc] + eifslope*exp((v[cc]-vth[cc])/eifslope))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C;
							end
							v[cc] += dt*dv;
							if v[cc] > vpeak
								spiked[cc] = true
								wadapt[cc] += bw_adapt
							end
						else
							dv = (vresti - v[cc])/taui + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C;
							v[cc] += dt*dv;
							if v[cc] > vth0
								spiked[cc] = true
							end
						end

						if spiked[cc] #spike occurred
							spiked[cc] = true;
							v[cc] = vreset;
							lastSpike[cc] = t;
							totalspikes[cc] += 1;
							totsp += 1;
							if totsp < spmax
								spiketimes[totsp,1] = tt; # time index as a sparse way to save spiketimes
								spiketimes[totsp,2] = cc; # cell id
							elseif totsp == spmax
								spiketimes[totsp,1] = tt; # time index
								spiketimes[totsp,2] = cc; # cell id

								totsp = 0 # reset counter total number of spikes
								# store spiketimes
								spikestore += 1
								#h5write(savefile, "dursimspikes/spiketimes$(spikestore)", spiketimes)
							end


							trace_istdp[cc] += 1.;
							if cc<=Ne
								x_vstdp[cc] += 1. / taux;
							end

							if cc <= Ne && thrchange # only change for excitatory cells and when thrchange == true
								vth[cc] = vth0 + ath;
							end

							#loop over synaptic projections
							for dd = 1:Ncells # postsynaptic cells dd  - cc presynaptic cells
								if cc <= Ne #excitatory synapse
									forwardInputsE[dd] += weights[cc,dd];
								else #inhibitory synapse
									forwardInputsI[dd] += weights[cc,dd];
								end
							end

						end #end if(spiked)
					end #end if(not refractory)

					if ifiSTDP # select if iSTDP
						#istdp
						if spiked[cc] && (t > stdpdelay)
							if cc <= Ne #excitatory neuron fired, potentiate i inputs
								for dd = (Ne+1):Ncells
									if weights[dd,cc] == 0.
										continue
									end
									weights[dd,cc] += eta*trace_istdp[dd]

									if weights[dd,cc] > Jeimax
										weights[dd,cc] = Jeimax
									end
								end
							else #inhibitory neuron fired, modify outputs to e neurons
								for dd = 1:Ne
									if weights[cc,dd] == 0.
										continue
									end

									weights[cc,dd] += eta*(trace_istdp[dd] - 2*r0*tauy)
									if weights[cc,dd] > Jeimax
										weights[cc,dd] = Jeimax
									elseif weights[cc,dd] < Jeimin
										weights[cc,dd] = Jeimin
									end
								end
							end
						end #end istdp
					end # ifiSTDP

					if tripletrule

					#triplet, ltd component
					if spiked[cc] && (t > stdpdelay) && (cc <= Ne)
						r1[cc] = r1[cc] + 1 # incrememt r1 before weight update
						for dd = 1:Ne #depress weights from cc to all its postsyn cells
							# cc = pre dd = post
							if weights[cc,dd] == 0. # ignore connections that were not establishe in the beginning
								continue
							end

			                weights[cc,dd] -= o1[dd]*(A_2m + A_3m*r2[cc])

							if weights[cc,dd] < Jeemin
								weights[cc,dd] = Jeemin
							end

						end # for loop over Ne
						 r2[cc] = r2[cc] + 1 # increment after weight update
					end # ltd

					#triplet, ltp component
					if spiked[cc] && (t > stdpdelay) && (cc <= Ne)
						o1[cc] = o1[cc] + 1 # incrememt r1 before weight update
						# cc = post dd = pre
						for dd = 1:Ne #increase weights from cc to all its presyn cells dd
							if weights[dd,cc] == 0.
								continue
							end

							weights[dd,cc] += r1[dd]*(A_2p + A_3p*o2[cc]) #A_2p = 0

							if weights[dd,cc] > Jeemax
								weights[dd,cc] = Jeemax
							end

						end # loop over cells presynaptic
						o2[cc] = o2[cc] + 1 # increment after weight update

					end #ltp
        #=
				else # not triplet but voltage rule

				#vstdp, ltd component
					if spiked[cc] && (t > stdpdelay) && (cc < Ne)
						for dd = 1:Ne #depress weights from cc to cj
							if weights[cc,dd] == 0.
								continue
							end

							if u_vstdp[dd] > thetaltd
								weights[cc,dd] -= altd*(u_vstdp[dd]-thetaltd)

								if weights[cc,dd] < Jeemin
									weights[cc,dd] = Jeemin

								end
							end
						end
					end #end ltd

					#vstdp, ltp component
					if (t > stdpdelay) && (cc <= Ne) && (v[cc] > thetaltp) && (v_vstdp[cc] > thetaltd)
						for dd = 1:Ne
							if weights[dd,cc] == 0.
								continue
							end

							weights[dd,cc] += dt*altp*x_vstdp[dd]*(v[cc] - thetaltp)*(v_vstdp[cc] - thetaltd)
							if weights[dd,cc] > Jeemax
								weights[dd,cc] = Jeemax
							end
						end
					end #end ltp
        =#
				end # if triplet rule
			end #end loop over cells
			forwardInputsEPrev = copy(forwardInputsE)
			forwardInputsIPrev = copy(forwardInputsI)

			# once the actual stimulation begins, eventually readjust the learning rate
			if tt == ttstimbegin

						tau_p = 16.8;        # in ms
						tau_m = 33.7;        # in s
						tau_x = 101.0;        # in s
						tau_y = 125.0;        # in s
						# init LTP and LTD variables
						A_2p = adjustfactor*7.5*10^(-10); # pairwise LTP disabled
						A_2m = adjustfactor*7.0*10^(-3);  #small learning rate
						A_2m_eff = A_2m;       #effective A_2m, includes the sliding threshold
						A_3p = adjustfactor*9.3*10^(-3)
						A_3m = adjustfactor*2.3*10^(-4); # triplet LTP disabled

						eta_old = eta
						eta = adjustfactorinhib*eta_old
						println("new eta iSTDP",eta)
            #=
						if T > 1
							h5write(savefile, "params/TripletTausAs_stim", [tau_p,tau_m, tau_x,tau_y , A_2p , A_2m, A_3p, A_3m])
							h5write(savefile, "params/eta_stim", eta)

							h5write(savefile, "params/adjustfactor2", adjustfactor)
							h5write(savefile, "params/adjustfactorinhib2", adjustfactorinhib)

							h5write(savefile, "dursim/weights", weights)

						end # if T > 1
            =#
			end # if tt == ttstimbegin
		end # tt loop over time
	print("\r")

	println("simulation finished")

	return totalspikes
end # function simulation


##

weights_start = copy(weights)

myT = Int64(10E3)

@time totalspikes = runsimulation_inhibtuning(ifiSTDP,ifwadapt,stimparams,stimulus, weights,
  assemblymembers, spiket, storagedec, storagetimes,savefile, lenpretrain, inhibassemblies,
  adjustfactor = adjustfactor, adjustfactorinhib = adjustfactorinhib,
  inhibfactor = inhibfactor, T = myT )

totalrates = totalspikes ./ myT

weights_end = copy(weights)

# ---------------------- store final params ---------------------------------------------------------
#=
h5write(savefile, "postsim/weights", weights)
h5write(savefile, "postsim/spiketimes", spiket)
h5write(savefile, "postsim/totalspikes", totalspikes)
=#

# start evaluation
# include("initevalimmediate.jl")
##

# ---------------------- DO THINGS ---------------------------------------------------------

function _rediff(a,b)
  if a+b == 0.0
    return 0.0
  else
    return (a-b)/(0.5(a+b))
  end
end

_ = let ws = weights_start, wn=weights_end
  di = wn .- ws 
  direl  = map(ww->_rediff(ww...),zip(wn,ws))
  # extrema(we[1:Ne,1:Ne] .- ws[1:Ne,1:Ne])
  #@show extrema(di[Ne+1:end,1:Ne])
  # heatmap(di[Ne+1:end,1:Ne])
  @show extrema(di)
  heatmap(di)
end

bar(totalspikes)
histogram(totalspikes)

## ---------------------- BUILD EQUIVALENT NETWORK -----------------------------------------



		# taue = 20 #e membrane time constant ms
		# taui = 20 #i membrane time constant ms
		# vreste = -70 #e resting potential mV
		# vresti = -62 #i resting potential mV
		# vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset mV
		# eifslope = 2 #eif slope parameter mV
		# C = 300 #capacitance pF
		# erev = 0 #e synapse reversal potential mV
		# irev = -75 #i synapse reversal potntial mV
		# vth0 = -52 #initial spike voltage threshold mV
		# thrchange = false # can be switched off to have vth constant at vth0
		# ath = 10 #increase in threshold post spike mV
		# tauth = 30 #threshold decay timescale ms
		# vreset = -60 #reset potential mV
		# taurefrac = 1 #absolute refractory period ms

		# synaptic kernel
		# tauerise = 1 #e synapse rise time
		# tauedecay = 6 #e synapse decay time
		# tauirise = .5 #i synapse rise time
		# tauidecay = 2 #i synapse decay time

		# external input
		# rex = 4.5 #external input rate to e (khz) since the timestep is one ms an input of 4.5 corresp to 4.5kHz
		# rix = 2.5#2.25 #external input rate to i (khz) # on average this is reduced to 2.25
    # initial rix 2.5
    # reduced rix 2.3044
		# Jex = 1.78 #external to e weight pF
		# Jix = 1.27 #external to i weight pF


# push!(LOAD_PATH, abspath(@__DIR__,".."))

using SpikingRNNs; const global S = SpikingRNNs
using SparseArrays

dt = 0.1E-3
myτe = 20E-3 # seconds
myτi = 20E-3 # seconds
τrefr= 1E-3 # refractoriness
vth_e = 20.   # mV
vthexp = -52.0 # actual threshold for spike-generation
vth_i = vthexp
eifslope = 2.0
Cap = 300.0 #capacitance mF
v_rest_e = -60.0
v_rest_i = -60.0
v_rev_e = 0.0
v_rev_i = -75.0
v_leak_e = v_rest_e
v_leak_i = v_rest_i
v_reset_e = v_rest_e
v_reset_i = v_rest_i

# synaptic kernel
tauerise = 1E-3 #e synapse rise time
tauedecay = 6E-3 #e synapse decay time
taueplus,taueminus  = tauedecay, tauerise
tauirise = 0.5E-3 #i synapse rise time
tauidecay = 2E-3 #i synapse decay time
tauiplus,tauiminus = tauidecay,tauirise

# input parameters

in_rate_e = 4.5E3
in_rate_i = 2.3044E3
Jin_e = 1.78
Jin_i = 1.27

## 
nt_e = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SpikeGenEIF(vthexp,eifslope)
  S.NTLIFConductance(sker,sgen,myτe,Cap,
   vth_e,v_reset_e,v_rest_e,τrefr,v_rev_e)
end
ps_e = S.PSLIFConductance(nt_e,Ne)

nt_i = let sker = S.SKExpDiff(tauiplus,tauiminus)
  sgen = S.SpikeGenNone()
  S.NTLIFConductance(sker,sgen,myτi,Cap,
   vth_i,v_reset_i,v_rest_i,τrefr,v_rev_i)
end
ps_i = S.PSLIFConductance(nt_i,Ni)

## static inputs

nt_in_e = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoisson(in_rate_e)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
ps_in_e = S.PSInputPoissonConductance(nt_in_e,Jin_e,Ne)

nt_in_i = let sker = S.SKExpDiff(taueplus,taueminus)
  sgen = S.SGPoisson(in_rate_i)
  S.NTInputConductance(sgen,sker,v_rev_e) 
end
ps_in_i = S.PSInputPoissonConductance(nt_in_i,Jin_i,Ni)



## connections 
conn_ee = let w_ee = weights[1:Ne,1:Ne]
  S.ConnGeneralIF2(sparse(w_ee))
end
conn_ii = let w_ii = weights[Ne+1:end,Ne+1:end]
S.ConnGeneralIF2(sparse(w_ii))
end
conn_ei = let w_ei = weights[1:Ne,Ne+1:end]
  S.ConnGeneralIF2(sparse(w_ei))
end
conn_ie = let w_ie = weights[Ne+1:end,1:Ne]
  S.ConnGeneralIF2(sparse(w_ie))
end

## Populations

pop_e = S.Population(ps_e,(conn_ee,ps_e),(conn_ei,ps_i),
  (S.FakeConnection(),ps_in_e))

pop_i = S.Population(ps_i,(conn_ii,ps_i),(conn_ie,ps_e),
  (S.FakeConnection(),ps_in_i))
##
# that's it, let's make the network
myntw = S.RecurrentNetwork(dt,pop_e,pop_i)

# record spiketimes and internal potential
krec = 1
n_e_rec = 1000
n_i_rec = 1000
t_wup = 0.0
rec_state_e = S.RecStateNow(ps_e,krec,dt,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_state_i = S.RecStateNow(ps_i,krec,dt,Ttot;idx_save=collect(1:n_i_rec),t_warmup=t_wup)
rec_spikes_e = S.RecSpikes(ps_e,5.0,Ttot;idx_save=collect(1:n_e_rec),t_warmup=t_wup)
rec_spikes_i = S.RecSpikes(ps_i,5.0,Ttot;idx_save=collect(1:n_i_rec),t_warmup=t_wup)

## Run

times = (0:myntw.dt:Ttot)
nt = length(times)
# clean up
S.reset!.([rec_state_e,rec_spikes_e])
S.reset!.([rec_state_i,rec_spikes_i])
S.reset!.([ps_e,ps_i])
S.reset!(conn_ei)
# initial conditions
ps_e.state_now .= v_start[1:Ne]
ps_i.state_now .= v_start[Ne+1:end]

@time begin
  @showprogress 5.0 "network simulation " for (k,t) in enumerate(times)
    rec_state_e(t,k,myntw)
    rec_state_i(t,k,myntw)
    rec_spikes_e(t,k,myntw)
    rec_spikes_i(t,k,myntw)
    S.dynamics_step!(t,myntw)
  end
end

#S.add_fake_spikes!(1.0vth_e,rec_state_e,rec_spikes_e)
#S.add_fake_spikes!(0.0,rec_state_i,rec_spikes_i)
##

rates_e = let rdic=S.get_mean_rates(rec_spikes_e,dt,Ttot)
  ret = fill(0.0,n_i_rec)
  for (k,v) in pairs(rdic)
    ret[k] = v
  end
  ret
end
rates_i = let rdic=S.get_mean_rates(rec_spikes_i,dt,Ttot)
  ret = fill(0.0,n_i_rec)
  for (k,v) in pairs(rdic)
    ret[k] = v
  end
  ret
end


_ = let plt=plot(;leg=false),
  netest = n_e_rec
  scatter!(rates_e,rates_test[1:netest];ratio=1,
    xlabel="Dylan's model",ylabel="Suchulz et al")
  plot!(plt,identity; linewidth=2)
end

_ = let plt=plot(;leg=false),
  ntest = n_i_rec
  scatter!(rates_i,rates_test[Ne+1:Ne+ntest];ratio=1,
    xlabel="Dylan's model",ylabel="Suchulz et al")
  plot!(plt,identity;linewidth=2)
end

##


_ = let neu = 202,
  plt=plot()
  plot!(plt,rec_state_e.times,rec_state_e.state_now[neu,:];linewidth=2,leg=false,
    xlims=(0,1),ylims=(-70,-40))
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[neu,:];linewidth=2,linestyle=:dash)
end
_ = let plt=plot(),neu=16
  plot!(plt,rec_state_i.times,rec_state_i.state_now[neu,:];linewidth=2,leg=false,
    xlims=(0,1))
  ts = (1:size(vtest,2)).*0.1E-3
  plot!(plt,ts,vtest[Ne+neu,:];linewidth=2,linestyle=:dash)
end
