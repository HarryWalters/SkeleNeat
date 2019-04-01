from osim.env import ProstheticsEnv
from SkeleNeat import *
import gym
import pickle


# NOTE: This 'cousin' version of the DemoScript takes a slightly different approach to network complexification (1/10/18)
# at the end of a batch/generation, the program will have a chance of being 'complexified'.
# If a gene is 'complexified', it will be given 1 more node AND several more connections; 
# there'll be the same number of connections as there are inputs
# This is intentional! The network doesn't start out as being fully connected! this means that there'll always be nodes to connect to!
# I reckon I'll make the 'complexification' probability as the product of the newNode and newCon probabilities.

# Rough pseudo code:
# make a copy of the breed function and call it complexify
# Have your temporary repopulation list
# for the percentage of non-mated genes (I think it's 25%?)
	# if they are chosen to be complexified (per our complexProbab)
		# add one node to the currentGenome
		# add len(currentGenome.sensor)-1 new connections to the currentGenome 
		# the -1 is because when you add a node, you superimose it onto a previous connection

#IMPORTANT
#env = ProstheticsEnv(visualize=True)
#observation = env.reset()
# This list of 22 continuous values correspond to muscle exitations; this is the output of my controller
DemoAction = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# added extra line
#print(action)


# this function converts the observation dictionary data into a list of 412 elements (4/10/18)
# Source: https://stackoverflow.com/questions/39135433/how-to-flatten-nested-python-dictionaries

def flatten(d):
	res = []  # Result list
	if isinstance(d, dict):
		for key, val in sorted(d.items()):
			res.extend(flatten(val))
	elif isinstance(d, list):
		res = d        
	else:
		res = [d]
	return res


# Memory; use the last 50 frames? last 5 seconds? idk of data  as the unput
# When youu start, the last n seconds will all be 0?, and the latest timestep will be the first 'frame' of memory
# once the 5 seconds or whatever of data are used up, the system overwrites the olderst frame and writes the newest frame to the front
# THE NETWORKS INPUT IS MEMORY!!!

# THIS MEANS, I can use a multilayer perceptron or otehr traditional neural networks!!!
# howevewr, this means im gonna have a massive number of inputs, a massive number of of nodes? maybe?
# That;s why mtyy genetic algortihm is so great; because it is a dynamic algorithm that slowly grows the network
simSpecies = species()

# for uniformly peterbing the weights
lowWeight = -1
highWeight = 1
# The number of decimal places 
decPlace = 2
# larger to make room for the bigger c3
#deltaT = 3.0 # default? 
#deltaT = 0.1
#deltaT = 4.0
deltaT = 4.0
c1 = 1.0
c2 = 1.0
# larger to allow for finer distinctions between species based on weight differences
c3 = 3.0
#c3 = 2.0
# If the maximum fitness of a species did not improve in 15 generations, the networks in the stagnant species were not allowed to reproduce
#stagSpeciesThreshold = 15
stagSpeciesThreshold = 15
# The champion of each species with more than 5 networks was copied into the next generation unchanged
champGenomeThreshold = 5
# There's an 80% chance of a genome having its connection weights mutated
conWeightMutProb = 0.8
# In which case each weight had a 90% chance of being uniformly perturbed
conWeightPeterbProb = 0.9
# and a 10% chance of being assigned a new random value
conWeightRandomProb = 1 - conWeightPeterbProb
conNodeMutProb = 0.03
# There was a 75% chance that an inherited gene was disabled if it was disabled in either parent
childDisabProb = 0.75
# In each generation, 75% of offspring resulted from mutation with crossover
#mutWithCrossover = 0.75
# experimental?
mutWithCrossover = 0.50
#mutWithCrossover = 0.9
# In each generation, 25% of offspring resulted from mutation without crossover
mutWithoutCrossover = 1 - mutWithCrossover
interspeciesMatingRate = 0.001
#interspeciesMatingRate = 1

# 28/08/18
# What if I could use NEAT to predict the fitness value for a given network?
# a rough idea would be the 158 inputs from the system; the hidden layers could be reccurent/ store 'memory'?
# I could test it on past observations and their fitness values
# I could then apply the trained network in this simulation;
# If the predictor network think's it's gonna fail, stop the program?
# It would need to have some sort of confidence level for each prediction though.

# [[observation],[fitness value]]
predictor = [[],[]]
#[[[69.420,911.59],[12.11,4.22]],[-55,106]]
# you can see how each element corresponds to each example
# predictor[0] is [69.42,911.59],[12.11,4.22]
# predictor[1] is [-55,106]
# our fitness value would be the actual value - predicted value?
# maybe that would be the penalty.
# We could have a (arbitrary?) value of 100 and penalise the abs(actual-predicted)
# update (9/03/19)
# added an action list; to be used alongside the predictor list
actionList = [[]]


#for i in range(200):
# a 'run'
#simAmount = 3
#populationNum = 5
#simAmount = 40
# it takes about 8-9 seconds to run the 'dumb' genomes on my laptop
# so a population of 50 takes 7 minutes
# doing that 40 times should take around 4.7 hours? not accounting for different sized species or complexities
# when I go above 25 simulations, it just stops (while it builds the new one)
#populationNum = 50

#NOTE: created a bestGenomeList for diagnostics. Don't know wh I didn't think of this earlier (26/09/18)
bestGenomeList = []
#simAmount = 1000
simAmount = 1000
# NGL, I might need to have a system that randomly adds new genomes into the system (if any get culled/killed)
#populationNum = 500
populationNum = 50
# The number of nodes in the hidden layer, when starting a population out.
hiddenNum = 0
population = []
# for debug (30/08/18)
populationLengths = []
speciesSizes = []
env = ProstheticsEnv(visualize= True)
observation = env.reset(project = False)
observation = flatten(observation)
# This is the current generation we are looking at.
currentGeneration = 0
for i in range(0,simAmount):
	#inefficient
	currentGeneration = i
	# if this is our first time running through the algorithm,
	if i == 0:
		# reset the environment ('good practice'?)
		#observation = env.reset(project=True)
		observation = env.reset(project= False)
		# uses that previously made function to convert the dictionary into a list of 412 elements (4/10/18)
		observation = flatten(observation)
		# add j amount of random 'dumb' genomes to the population
		for j in range(0,populationNum):
			# Note: trying out adding some hidden layers (12/09/18)
			population.append(copy.copy(genome([len(observation),hiddenNum,22])))

		print('the population is: ', population)
		populationLengths.append(len(population))
	# if this isn't our first time running through the algorithm
	else:
		# reset the population list
		population = []
		print('we are choosing from species', simSpecies.speciesList)
		# for each species
		for j in range(0, len(simSpecies.speciesList)):
			# Note: changing -1 to -2, as we've added in a blank list (13/09/18)
			print('adding in ', simSpecies.speciesList[j][len(simSpecies.speciesList[j])-1])
			# add the genomes from the latest generation of the jth species to the population list
			population.append(simSpecies.speciesList[j][len(simSpecies.speciesList[j])-1])
			#print('test statement: population is ', population)
			# ugly line, but append makes a list in a list

		# NOTE:
		# Super ugly addition, but it solves our problems where only the frist species gets a fitness value. (3/09/18)
		# Make a copy of the current populations list
		tempPopulations = copy.copy(population)
		# merge all of the values together.
		# please excuse the ugliness
		population = []
		for j in range(0,len(tempPopulations)):
			population  = population+tempPopulations[j]

		#print('possible debug: population before editing is ', population)
		#population = population[0]
		print('the new population is of length: ', len(population))
		populationLengths.append(len(population))

		# get the values from the latest generation of each species and put them in the population
	# for each of the j genomes in the population,
	# If there's no more genomes
	#if len(population) == 0:

	# we can only crossBreed 2 genomes together
	if len(population) < 2:
		print('I think all species are extinct')
		# add *popNum* more to be simulated
		for j in range(0,populationNum):
			population.append(copy.copy(genome([len(observation),hiddenNum,22])))
		#print(population)
		#simSpecies.display()
	for j in range(0, len(population)):
		# If a network get's stuck/ if it takes more than *30* seconds to do a step; end the simulation
		# Note the +1: python starts at 0 but humans start counting from 1. (2/09/18)
		# Also note how the current population length is taken from the populationLengths list; it's just some simple housekeeping
		print('begin simulation number ', j+1, ' out of ', populationLengths[len(populationLengths)-1],'. Generation number ', i+1)
		print('The current population lengths are: ', populationLengths)
		print('The list of species for each generation is: ', speciesSizes)
		#print('There are ', len(self.speciesList), ' different species.')
		# put all our obervation/fitness values in a new list
		predictor[0].append([])
		predictor[1].append([])

		actionList[0].append([])

		# make the jth genome in the population the simulated genome
		simGenome = population[j]

		#env = ProstheticsEnv(visualize=True)
		#env = ProstheticsEnv(visualize= True)
		#observation = env.reset(project=True)
		observation = env.reset(project = False)
		observation = flatten(observation)
		#print('The observation list is of length', len(observation))
		#print(observation)
		#print(len(observation))
		#simulating genome (the genome we're putting through the simulation)

		# if this is our first time running the simulation (or a species has been culled? not sure),
		# use random genomes
		# if i == 0:
		# 	simGenome = genome([len(observation),0,22])
		#for i in range(0,1):
		#	simGenome.addNode()
		#for i in range(0,50):
		#	simGenome.addCon()
		#DemoAction = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0]
		#DemoAction = simGenome.buildNetwork(DemoAction,'sig',4.9)
		total_reward = 0.0
		index = 0
		fall = False
		# NOTE: added it to 20 seconds (20/9/18)
		while index < 2000 and fall == False:
		# when I've got a controller, it will output 'action', a list of continuous values
		# in [0,1]. This list will be 22 elements or 19 elements long (with prosthetic and without)
		# step(action, project = True)
		#observation, reward, done, info = env.step(env.action_space.sample())
			print('we are at step ', index, '. Applying the last 412 data points to the 22 muscle excitations')
			#DemoAction = simGenome.buildNetwork(DemoAction,'sig',4.9)
			#observation, reward, done, info = env.step(simGenome.buildNetwork(observation,'sig',4.9))

			# add our observation to the ith element of predictor[i]
			predictor[0][len(predictor[0])-1].append(observation)
			action = simGenome.buildNetwork(observation,'tanh', 1)
			actionList[0][len(actionList[0])-1].append(action)

			[observation, reward, done, info] = env.step(action, project = False)
			observation = flatten(observation)
			total_reward += reward
			# if the simulation is done
			if done:
			#break
				fall = True
				# place the genome in the species
				# Note: trying something out (I added +1). This is to prevent a problem where all simulated genomes are pooled in the same generation as the parents; which can cause exponential growth. (13/09/18)
				simGenome.generationNum = i
				simGenome.fitness = total_reward
				# add our final fitness value to the ith element of predictor[1]
				predictor[1][len(predictor[1])-1].append(total_reward)
				#print('predictor is:')
				#print(predictor)
				print('simGenome generationNum is ', simGenome.generationNum)
				print('type of simGenome: ', type(simGenome))
				print('type of simGenomeCopy: ', type(copy.copy(simGenome)))
				simSpecies.place(copy.copy(simGenome), deltaT, c1, c2, c3)
		# Every time step, add 100  to each element in the list
		#DemoAction = [i+0.01 for i in DemoAction]
			#total_reward_previous = total_reward
			index += 1

			#if total_reward_previous > total_reward:
			#	theBestGene = copy.copy(myNameJeff)
			#	theBestReward = copy.copy(total_reward_previous)
	print('we are at the end of generation ', i)
	print('the current population lengths are: ', populationLengths)
	# reset the current innovation list
	glob.currInnovList = []
	simSpecies.display()

	# I commented this out (29/08/18)
	# I might need to add a -1 somewhere, as the latest generation won't have fitness values yet. (29/08/18)

	# # If we're on our first run,
	# if i == 0:
	# 	# set it to the best one from the first species
	#theBestCurrentGenome = simSpecies.largestFitness(0,2)
	theBestCurrentGenome = simSpecies.largestFitness(0,1)
	# # for each other species
	for k in range(1,len(simSpecies.speciesList)):
		#if simSpecies.largestFitness(k,2).fitness > theBestCurrentGenome.fitness:
		if simSpecies.largestFitness(k,1).fitness > theBestCurrentGenome.fitness:	
	# 		# the best genome in the ith species is the new 'best gene'
			#theBestCurrentGenome = simSpecies.largestFitness(k,2)
			theBestCurrentGenome = simSpecies.largestFitness(k,1)
	# NOTE the copy function. This is to try solve an error where values that had already been appended were being changed. (29/09/18)
	bestGenomeList.append(copy.copy(theBestCurrentGenome))
	# # makes a .ser file of the best current gene
	with open("bestCurrentGenome.ser", "wb") as dataFile:
		pickle.dump(theBestCurrentGenome, dataFile)

	with open("bestGenomeList.ser", "wb") as dataFile:
		pickle.dump(bestGenomeList, dataFile)

	with open("currentSpeciesList.ser", "wb") as dataFile:
		pickle.dump(simSpecies.speciesList, dataFile)

	with open("predictorDataset.ser", "wb") as dataFile:
		pickle.dump(predictor, dataFile)

	with open("actionDataset.ser", "wb") as dataFile:
		pickle.dump(actionList, dataFile)

	##### SIMULATION RUN END
	# cull any species that haven't improved in 15 generations
	# I should make it so the argument is the generation, not the entire genome
	simSpecies.cull(simGenome, stagSpeciesThreshold)
	# if in the 80% chance the connection weights will be mutated,
	# if random.choices([0,1], weights = [1-conWeightMutProb, conWeightMutProb])[0] == 1:
	# 	if random.choices([0,1], weights = [1-conWeightPeterbProb, conWeightPeterbProb])[0] == 1:
	# 		# weights are uniformly peterbed by some specified range

	# 		# for each 
	# 		#weight += round(random.uniform(lowValue,highValue), roundedPlace)
	# 	else:
	# 		print('working on it')
	# 		# assigned a new random value

	# 75 percent of offspring are from crossover
	#mutWithCrossover = 0.75
	#mutWithCrossover = 0.9
#stagSpeciesThreshold = 15
# The champion of each species with more than 5 networks was copied into the next generation unchanged
	#champGenomeThreshold = 5
# There's an 80% chance of a genome having its connection weights mutated
	weightMutProb = 0.8
# In which case each weight had a 90% chance of being uniformly perturbed
	peterbWeightProb = 0.9
# and a 10% chance of being assigned a new random value
	randWeightProb = 1 - peterbWeightProb

# 3% chance of having a new node
	newNodeProb = 0.03
# NOTE: Trying out .3% chance of adding node. (17/09/18)
	#newNodeProb = 0.003
	# NOTE: trying out half the value from last time (20/9/18)
	#newNodeProb = 0.0015
# I am experiencing an issue where the genomes become more complex, but they don't perform very well. Genomes become more complicated faster than they improve
# I also recognise that there could be an error if a genome doesn't have any more nodes to connect to. (newCon prob might also have to be smaller than the default .3?) 

# in small networks, 5% chance of having a new connection
	#newConProb = 0.05
	# NOTE: trying out half the value from last time (20/9/18)
	#newConProb = 0.025
# but in big networks, 30% chance of having a new connection
	newConProb = 0.3
	#newConProb = 0.0

	# a new probability. If it's used (>0), the newNodeProb and newConProb get set to 0
	#complexifyProb = newConProb * newNodeProb
	complexifyProb = 0.33
	# if a genome is complexified, 10 new connections will be added
	# note the 26: my best performing models added 10 connections. The updated genomes use 2.6 more nodes
	# newConNum = 26
	newConNum = 10

# There was a 75% chance that an inherited gene was disabled if it was disabled in either parent
	#childDisabProb = 0.75
# In each generation, 25% of offspring resulted from mutation without crossover
	#mutWithoutCrossover = 0.25
	interSpeciesMatingRate = 0.001

	simSpecies.display()
	# the 0th species in simSpecies
	#simSpecies.champGenome(0,champGenomeThreshold)


	# the BIG papa
	simSpecies.breed(populationNum, mutWithoutCrossover, weightMutProb, peterbWeightProb, randWeightProb, complexifyProb, newConNum, newConProb, newNodeProb, childDisabProb, champGenomeThreshold, interSpeciesMatingRate, currentGeneration)
	# add the number of unique species to the speciesSizes list at the end of a generation/simulation
	speciesSizes.append(len(simSpecies.speciesList))
	#if simSpecies.breed(mutWithoutCrossover, weightMutProb, peterbWeightProb, randWeightProb, newConProb, newNodeProb, childDisabProb, champGenomeThreshold, interSpeciesMatingRate, currentGeneration) == "extinct":
		# we have no more usable species.
		# generate *populationNum* amount of new species to test
	#	print('extinct. heck')
print('genome count: ', simSpecies.genomeCount(0))
simSpecies.display()
print(len(simSpecies.speciesList))
#for i in range(0, len(simSpecies.speciesList)):
#	for j in range(0, len(simSpecies.speciesList[i])):
#		for k in range(0, len(simSpecies.speciesList[i][j])):
#			simSpecies.speciesList[i][j][k].display()
#print(total_reward)


print('end')
print('the population lengths are: ', populationLengths)
# some arbitrary value
theBestGene = simSpecies.largestFitness(0)
for i in range(1, len(simSpecies.speciesList)):
	# if another species has a better fitness value
	if simSpecies.largestFitness(i).fitness > theBestGene.fitness:
		# the best genome in the ith species is the new 'best gene'
		theBestGene = simSpecies.largestFitness(i)
	print('the best fitness for species ', i,' is ', theBestGene.fitness)

print('the best gene comes from generation number: ', theBestGene.generationNum)
print('the best reward: ', theBestGene.fitness)
print('pickling the best gene')
# makes a .ser file of the best gene
with open("bestGenome.ser", "wb") as dataFile:
	pickle.dump(theBestGene, dataFile)
	# don't have to do this line, might be good practice?
	#dataFile.close()
with open("predictorDataset.ser", "wb") as dataFile:
	pickle.dump(predictor, dataFile)

with open("actionDataset.ser", "wb") as dataFile:
	pickle.dump(actionList, dataFile)
	# added extra line
	# there are 158 elements in 'observation'
	# 'reward' is a float, indicating how far in front the pelvis is.
	# We want to miximise 'reward'
	# 'done' a boolean which *i believe* returns true when the pelvis is lower than
	# 0.6m or more than 10000 iterations were reached
	#I dont think I need to worry about it for the NEAT algorithm
	#if (i == 50|100|150|200):
	#	print(observation)
	#	print(reward)
	#	print(done)
	#	print(info)