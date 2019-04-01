# SkeleNeat

# By Harry Walters 
# First made 25/06/2018

# My attempt at a NEAT implementation for the 2018 crowd AI challenge
# (Building a controller for a bipedal skeleton to use)

import numpy as np
# I'll probably just use math, instead of numpy, but we'll see what happens.
# I like using base packages.
import math
import random
# for building a network
import copy

# There needs to be a way to build a genome from 'linear representations'
# of network connectivity

# I might need to make a function that converts any input 
# (like a list of muscle activations): [value1,value2,value4,value4]
# into this form [[value1,1],[value2,1],[value3,1],[value4,1]]

class glob():
	#our global innovation number (27/08/18)
	globalInnovNum = 0
	# our current innovation list for that generation
	currInnovList = []
class genome():
	"""I'm still working on it!"""

	# Every Genome has a list of nodes and connections
	# if nothing is specified, the Node and Connection genes will be empty
	# rand is a boolean, if it's 1 AND nodeGenes/conGenes are specified,
	# a network will be automatically generated
	def __init__(self, nodeGenes, conGenes = 'd', rand=False, lowWeight= -1 , largeWeight= 1, generationNum = 0, speciesNum = 0):
		# a genome starts out with 0 hidden layers
		# every input node is connected to any one of the genome's output
		# creates a new list for every genome
		self.rand = rand
		# This tag is used to find which generation/ species a genome belongs to
		self.generationNum = generationNum
		self.speciesNum = speciesNum
		if len(nodeGenes) == 3:
			for i in range(0,3):
				if nodeGenes[i] <0:
					nodeGenes[i] = 0
			self.nodeGenes = []
			[self.nodeGenes.append([i,1]) for i in range(1, nodeGenes[0] + 1)]
			[self.nodeGenes.append([i,2]) for i in range(1 + nodeGenes[0], nodeGenes[0] + nodeGenes[1]+1)]
			[self.nodeGenes.append([i,3]) for i in range(1 + nodeGenes[0] + nodeGenes[1], nodeGenes[0] + nodeGenes[1] + nodeGenes[2]+1)]
		else:
			self.nodeGenes = nodeGenes
		self.lowWeight = lowWeight
		self.largeWeight = largeWeight
		self.sensor = []
		self.hidden = []
		self.output = []
		# a list of just the node numbers
		self.nodeNumList = [i[0] for i in self.nodeGenes]

		self.nodeCount = len(self.nodeGenes)

		if conGenes == 'd':
			self.conGenes = []
		else: 
			self.conGenes = conGenes
		self.conCount = len(self.conGenes)

		# used for checking the last element (the type) of each node
		temp = [i[1] for i in self.nodeGenes]
		for i in range(0,self.nodeCount):
			if temp[i] == 1:
				self.sensor.append([i[0] for i in self.nodeGenes][i])
			elif temp[i] == 2:
				self.hidden.append([i[0] for i in self.nodeGenes][i])
			elif temp[i] == 3:
				self.output.append([i[0] for i in self.nodeGenes][i])

		# A BIT SCRAMBLED, WILL WORK ON IT AFTER JAZZ
		if len(self.conGenes) == 0:
			self.conGenes = copy.copy(self.sensor)

		self.enabConGenes = []
		for i in range(0, self.conCount):
			if [i[3] == 1 for i in self.conGenes][i]:
				self.enabConGenes.append(self.conGenes[i])

		# set up
		#self.innovNum = 0
		self.innovNum = glob.globalInnovNum
		self.enabConCount = len(self.enabConGenes)


		# If there is an entry in nodeGenes but no entry in conGenes 
		if len(self.nodeGenes) > 0 and self.conCount == 0:
			#self.conGenes = sum(nodeGenes)-1
			#for i in range(0, sum(nodeGenes)-1):
			# for each sensor node
			for i in range(0,len(self.sensor)):
			# randomly connect it to an output node
				#print(i)
				#print(self.sensor[i])
				#print(self.output[i])
				self.addCon(self.sensor[i], random.choice(self.output))
				self.conGenes.remove(self.sensor[i])
				#print(self.conGenes)

			for i in range(0,len(self.hidden)):
			# randomly connect it to an output node
				#print(i)
				#print(self.sensor[i])
				#print(self.output[i])
				self.addCon(self.hidden[i], random.choice(self.output))
				#self.conGenes.remove(self.hidden[i])
				#print(self.conGenes)
				
			# if there are any output nodes with no inCon,
			# for each output node
			for i in range(0, len(self.output)):
				#print('i is: ',self.output[i])
				# if there are any output nodes with no inCon,
				if len(self.currInCons(self.output[i],self.conGenes)) == 0:
					#print('currInCons: ', self.currInCons(self.output[i],[self.output]))
					# randomly connect it to a sensor node
					self.addCon(random.choice(self.sensor), self.output[i])
					#print(self.conGenes)

		
		# extra objects for checking the size of the genes
		# a count of only the enabled connections
		#if self.conCount != 0:
		#	self.innovNum = max([i[4] for i in self.conGenes])
		#else: 
		#	self.innovNum = 0
			#self.enabConCount = 0
		# an integer, starts out empty
		self.fitness = 'empty'
		self.adjFitness = 'empty'

	def display(self):
		print("Node Genes: " + str(self.nodeGenes))
		print("Nodes: ")
		print("	Sensor: " + str(self.sensor))
		print("	Hidden: " + str(self.hidden))
		print("	Output: " + str(self.output))
		print("Connection genes: " + str(self.conGenes))
		print("Enabled connection genes: " + str(self.enabConGenes))
		print("Number of nodes: " + str(self.nodeCount))
		print("Number of connection genes : " + str(self.conCount))
		print("Number of enabled connections: " + str(self.enabConCount))
		print("Max innovation number: " + str(self.innovNum))

	def addCon(self, inCon='r', outCon='r', weight='r', enabled=1):
		#self, inCon = 0, outCon = 0, enabled = 1, weight):
		# if the user hasn't specified the weight of this new connection,
		# 'r' for random
		if enabled > 1:
			#raise neatError("Enabled can only be 0 or 1.")
			self.enabled = 1
		elif enabled < 0:
			#raise neatError("Enabled can only be 0 or 1.")
			self.enabled = 0
		elif type(enabled) != int:
			raise neatError("Enabled can only an integer (0 or 1).")

		if weight == 'r':
			# The weight of this new connection will be randomly generated
			self.weight = self.randWeight()
		# If the user gives an integer for a weight
		elif type(weight) == int:
			# it will be converted to a float
			self.weight = float(weight)
		else:
			self.weight = weight

		if inCon == 'r' or outCon == 'r':
			randInCon = self.randInCon()
			randOutCon = self.randOutCon(randInCon)
			#print('node ', randInCon)
			#print(' goes to node ', randOutCon)

		if inCon == 'r':
			self.inCon = randInCon
		elif type(inCon) != int:
			raise neatError("inCon must be an integer.")
		elif inCon < 1:
			raise neatError("inCon must be greater than 0.")
		else:
			self.inCon = inCon

		if outCon == 'r':
			self.outCon = randOutCon
		elif type(outCon) != int:
			raise neatError("outCon must be an integer.")
		elif outCon <1:
			raise neatError("outCon must be less than 0.")
		else:
			self.outCon = outCon

		#self.inCon = inCon
		#self.outCon = outCon
		# any given node will be a 2 element list
		# add that 2 element list to the general list
		# the output will be a list within list [[1,sensor],[2,input]
		##if type(nodeNum) != int or type(nodeType) != int:
			##raise neatError("The node number and type must both be integers.")
			##print("nodeType: 1 = Sensor, 2 = Hidden, 3 = Output")
		##elif nodeNum < 0 or nodeType < 0:
			##raise neatError("The node number and type must be greater than 0.")
		#a pythonic expression extracting only the 0th elements
		# change these temp variables?
		temp3 = [i[0] for i in self.nodeGenes]
		temp4 = [i[1] for i in self.nodeGenes]
		if outCon in self.sensor:
			raise neatError("outCon cannot be a sensor")
		# else if inCon not in the nodeGenes' inCons

		# ADDITION
		if inCon == 'r' or outCon == 'r':
			if randInCon not in temp3 and randInCon not in temp4:
				# Note, I', very sure there's a consistent problem about how child genomes don't inherit all of the parent nodes. (14/09/18)
				# raise neatError("inCon must be a node in the genome.")
				pass
			# else if outCon not in the nodeGenes' outCons
			elif randOutCon not in temp3 and randOutCon not in temp4:
				# raise neatError("outCon must be a node in the genome")
				pass
			for i in range(0,len(self.conGenes)):
				# if inCon and outCon are already connected, 
				if inCon == [i[0] for i in self.conGenes][i] and outCon == [i[1] for i in self.conGenes][i]:
					# NOTE: removed the 'error', so that the loop just skips this (14/09/18)
					# raise neatError("There is already a connection from " + str(inCon) + " to " + str(outCon) + ".")
					pass
		##elif nodeType > 3:
			##raise neatError("The node type can only be 1 = Sensor, 2 = Hidden or 3 = Output")
		
			# The total number of nodes increases by 1
			# maybe change conCount to some __init__ value
		self.conCount += 1
			# ADDITION
		#if enabled == 1:
		#	self.enabConCount += 1

		# check to see if self.inCon -> self.outCon (like, node 1 to node 4) already appears in the currInnovList (27/08/18)
		# currInnovList: [[[1,3],1],[[2,3],2],[[2,4],3],[[3,4],4],[[4,5],5]]
		# if it does, give our connection gene the same innovation number as seen in the currInnovList
		# else, increment the global innovation number, append the new connection (and it's updated globalInnovNum) to the currInnovList
		# ifend 
		# append our connection gene to the genome
		appears = False
		for i in range(0,len(glob.currInnovList)):
			# if our connection already appears in the currentInnovationList,
			if [self.inCon, self.outCon] == glob.currInnovList[i][0]:
				appears = True
				# give our connection gene the same innovNum as the original connection gene
				# [0] is because the output is a list, and we want the integer inside that list
				self.innovNum = glob.currInnovList[i][1]
		# if after the loop, our connection gene STILL doesn't appear,
		if appears == False:
			# increment the global innovation number
			glob.globalInnovNum += 1
			# append the new connection pairing + the new innovNum to the currInnovList
			glob.currInnovList.append([[self.inCon, self.outCon], glob.globalInnovNum])
			# set the connection gene's innovation number to the current global innovation number
			self.innovNum = glob.globalInnovNum
		#self.innovNum += 1
		self.conGenes.append([self.inCon, self.outCon, self.weight, enabled, self.innovNum])
			# ADDITION
		if enabled == 1:
			self.enabConCount += 1
			self.enabConGenes.append([self.inCon, self.outCon, self.weight, enabled, self.innovNum])
			# The innovation number increases by 1

	# Note that since I don't want a user messing with the innovNum,
	# I've made it an automatically updating integer in the object.
	##def addNode(self,inCon,outCon,weight,enabled=1):
		# if the 'in' connection comes from a non-existent node,


	# in adding a new node, the connection gene being split is disabled,
	# two new connection genes are added to the end of the genome
	# the new node is between the two new conenctions
	# a new node gene is added to the genome as well
	def addNode(self):
		# 'adding a node': note the 2, referring to a hidden node
		self.nodeCount += 1
		self.hidden.append(self.nodeCount)
		self.nodeGenes.append([self.nodeCount,2])
		# set a randomly selected enabled gene to temp
		temp = random.choice(self.enabConGenes)
		# worst comes to worst, I can remove a selected gene from the 
		# enabled set;
		self.enabConGenes.remove(temp) 
		self.enabConCount -= 1

		# NOTE: adding an extra line (26/09/18).
		# I recognise that even though the gene is disabled, it's not classified as disabled in the genomes conGenes list. 
		# To resolve that, I will remove the old 'enabled' version of that gene from the genome, disable it, and then re-insert it back into the list
		
		# NOTE: I've also added a temporary fix to a problem, where the temp gene would be in enabConGenes but not conGenes. 
		# This line checks to see if it's in conGenes before trying to remove it
		if temp in self.conGenes:
			self.conGenes.remove(temp)
		# disable that gene
		temp[3] = 0
		self.conGenes.append(temp)

		# randomly choose an enabled connection gene
		# disable that connection gene
		# add a connection from the old inCon to the new node
		# The new connections leading into the new node receives a weight of 1
		self.addCon(temp[0], self.nodeCount, 1.0)
		# add a connection from the new node to the old outCon
		# The new connection leadfing out receives the same weight as the old connection
		self.addCon(self.nodeCount, temp[1], temp[2])
		self.innovNum += 1

	# this returns all the outward connections from node inCon
	def currOutCons(self,inCon, conGenes = 'd'):
		# if this is left default
		if conGenes == 'd':
			# it will use the entire genome's connection genes
			conGenes = self.conGenes
		currOutConsList = []
		# If we know there are 0 connections, the system doesnt have to loop over every element 
		# I MIGHT CHANGE THIS LINE; this is all about how the genome is initialised
		# once a network is built, the connections will then be randomly added
		if len(conGenes) > 0:
			# a variable temporarily storing the input connections
			# this way, we only have to do this operation once
			temp = [i[0] for i in conGenes]
			for i in range(0,len(temp)):
				# if any of the connections come from inCon,
				if temp[i] == inCon:
					# add that outCon to a list
					currOutConsList.append([i[1] for i in conGenes][i])

		#if inCon in [i[0] for i in conGenes] == False:
			#There are no nodes that connect to
		#[i[0] == inCon for i in conGenes]
		return currOutConsList

	# Not sure if this will be used in the final version.
	def currDirOutCons(self, InCon, conGenes = 'd'):
		""" This function returns all nodes connected from node InCon that ARENT reccurent connections
		(i.e. nodes that connect to themselves)
		"""
		# if this is left default
		if conGenes == 'd':
			# it will use the entire genome's connection genes
			conGenes = self.conGenes
		# The non-recurrent connections will be added in a loop
		currOutConsList = self.currOutCons(InCon, conGenes)
		currDirOutConsList = []
		for i in range(0,len(currOutConsList)):
			# if any node doesnt connect to itself,
			if currOutConsList[i] != InCon:
				# add it to the list
				currDirOutConsList.append(currOutConsList[i])
		return currDirOutConsList	


	#[4,5],[4,6][6,4]: there are a total of 2 connections that eventually go into 4
	# returns all the connections that inputs 
	def allInCons(self, availableCons, conGene):
		availableConsList = []
		# if 4 goes to 5, extract 5
		targetOutCon = conGene[1]
		# For each of the available connection genes,
		for i in range(0,len(availableCons)):
			# if a given connection gene goes to
			# if 5 is an inCon of
			if targetOutCon == availableCons[i][1]:
				availableConsList.append(availableCons[i][1]) 

	# this returns all the nodes connected to node outCon
	# NOTE: this will also return disabled connected nodes
	# This is intended! the function randCon calls this function.
	def currInCons(self,outCon, conGenes = 'd'):
		""" This function returns all nodes connected to node outCon
		"""
		# if this is left default,
		if conGenes == 'd':
			# it will use the entire genome's connection genes
			conGenes = self.conGenes
		currInConsList = []
		if len(conGenes) > 0:
			# stores all the output connections in a temporary list
			temp = [i[1] for i in conGenes]
			for i in range(0,len(temp)):
				# if a connection goes to outCon,
				if temp[i] == outCon:
					# Add the connection number to a list
					currInConsList.append([i[0] for i in conGenes][i])
		return currInConsList

	def currDirInCons(self,outCon, conGenes = 'd'):
		""" This function returns all nodes connected to node outCon that ARENT reccurent connections
		(i.e. nodes that connect to themselves)
		"""
		# if this is left default,
		if conGenes == 'd':
			# it will use the entire genome's connection genes
			conGenes = self.conGenes
		# The non-recurrent connections will be added in a loop
		currInConsList = self.currInCons(outCon, conGenes)
		currDirInConsList = []
		for i in range(0,len(currInConsList)):
			# if any node doesnt connect to itself,
			if currInConsList[i] != outCon:
				# add it to the list
				currDirInConsList.append(currInConsList[i])
		return currDirInConsList

	# When a mutation occurs, this function will be called
	# It randomly selects an input node and a previously unconnected output node
	# it won't randomly connect to another sensor.

	# if all nodes are connected to every possible other node,
	# leave the network as is; return the original network from before randCon was invoked. 
	
	# total available outCons = total nodes - sensor nodes
	# total available outCons for a given node = total available outCons - a given node's outCons

	# if a node is conencted to every possiuble one, aka
	# if a node's OutCons equal the total available nodes

	# I might need to iteratively check using a for loop

	# totAvailInCons = #nodes that aren't connected to every possible one
	# tot AvailOutCons =  # all of the outCons - all sensors - any nodes already connected to the chosen inCon
	# Randomly choosing from the unconnected nodes
	# = random.choice(AvailInCon)
	# for all total available outCons:
			# 

	# in adding a new node, the connection gene being split is disabled,
	# two new connection genes are added to the end of the genome
	# the new node is between the two new conenctions
	# a new node gene is added to the genome as well
	def randNode(self):
		if self.nodeCount < 2:
			raise neatError("There must")

	# I NEED TO FINISH THIS
	def randWeight(self, weight = 0, decPlace = 2):
		# If the default is left,
		#if self.lowWeight == 'd':
			# set the low weight as -1
		#	self.lowWeight = -1
		# WHERE ARE WE GETTING self.largeWeight from??? I'd like to make it a global variable?
		#if self.largeWeight == 'd':
		#	self.largeWeight = 1
		# if the user doesn't specify, the range will be -1 and 1
		weight += round(random.uniform(self.lowWeight, self.largeWeight), decPlace)
		return weight
		# The weight will be an 

	# NOTE: trying out an old idea. I think it can work! (30/09/18)
	# instead of building a network from the ground up, I'll only work with the one's we got

	def isFullyConnected(self):
		# NOTE: I added this in case the user wanted to work with a fully connected genome (20/09/18)
		# a boolean that determines if a network has had all possible connections connected
		# Then, in the demo script, the user can specify:
		# while the genome isn't fully connected,
		# keep adding connections

		# if there aren't any possible nodes to connect, the outputted list will be of length 0 (cause 0 possible nodes)
		# therefore, this expression is a boolean that returns TRUE if a genome is fully connected.
		print('Trying this function out.')

		possNodes = [i[0] for i in self.nodeGenes]
			# for each element in the possible nodes
		for i in range(0,len(possNodes)):
				# if there are 0 possible
				# remove all the sensor nodes from the possible 
			totAvailOutCons = list(set([i[1] for i in self.conGenes]) - set(self.sensor) - set(self.currOutCons(i)))
			#print('total avail outcons:')
			#print(totAvailOutCons)
			if len(totAvailOutCons) == 0:
				possNodes.remove(i)
		return len(possNodes) == 0

	def randInCon(self):
		if self.nodeCount < 2:
			raise neatError("There must be at least 2 nodes to connect.")
			print("Have you built a network yet, using neatBuild?")
		# else, if a node is already connected to every possible node,
		# randomly choose any other node
		else:
			# The possible nodes to randomly select from will initialise with all of them
			# and then any 'full' nodes will be iteratively removed.
			possNodes = [i[0] for i in self.nodeGenes]
			# for each element in the possible nodes
			for i in range(0,len(possNodes)):
				# if there are 0 possible
				# remove all the sensor nodes from the possible 
				totAvailOutCons = list(set([i[1] for i in self.conGenes]) - set(self.sensor) - set(self.currOutCons(i)))
				if len(totAvailOutCons) == 0:
					possNodes.remove(i)
					#possNodes.remove(i)
			#for i in list(set(self.conGenes[0])):
			#	if len(set(totAvailOutCons)-set(outCons[i])) > 0:
			#		possNodes.append(i)

			# randomly choose from the available genes
			return random.choice(possNodes)
			#random.choice([availOutConsList])
			# return inCon
	def randOutCon(self, inCon):
		# copied from randInCon
		totAvailOutCons = list(set([i[1] for i in self.conGenes]) - set(self.sensor) - set(self.currOutCons(inCon)))
		# NOTE: added this line (18/09/18)
		# if there are no available connections to choose from, skip this step
		if len(totAvailOutCons) == 0:
			pass
		else:
			return random.choice(totAvailOutCons)

		#output will be outCon
		#randCon will randomly choose an inCon
		# from that inCon, randCon will then choose an outCon that
		# isn't connected to inCon
		#inCon = random.choice([i[0] for i in conGenes])
		# temporary variable storing the outputs of all conGenes
		#temp = [i[1] for i in conGenes]
		# remove any outCons already connected to inCon
		#temp.remove(inCon)
		# make a random choice from the remaining ones


# function for cross-breeding 2 genomes
# organisms in the same species must share the fitness of their niche (species?)

# Compatability distance genomes
##def delta(genome1,genome2,c1,c2,c3):
	# shared isnt used at the moment
	# the shared innov numbers (genome[2,0]) will randomly exchange genetic information
	##shared = set(genome1[2,0]) & set(genome2[2,0])
	##if genome1 > genome2:
	##	excess = set(genome1) - set(genome2)
	##	disjoint = set(genome2) - set(genome1)
	##else:
	##	excess = set(genome2) - set(genome1)
	##	disjoint = set(genome1) - set(genome2)

	##wBar = 
	# The number of genes in the larger genome
	##bigN = max()

# Genome1 and Genome2 are randomly selected
# remove self

	# The compatability distance
	# the large c3 is, the finer the distinctions between species based on weight differences
	# (The larger population has room for more species)
	# In all experiments, deltaT = 3.0
	# but to make room for the larger weight significance for a large c3, deltaT is 4.0

	# If a species did not improve in 15 generations,
	# the networks in the stagnant species were not allowed to reproduce.
	# The champion of each psecies with more than five networks is copied into the next generation, unchanged.
	# There is an 80% chance of a genome having its connection weights mutated,
	# If that happens, each weight has a 90% chance of being uniformly pertubed and a 10% chance of being assigned a new random value
	## Make the peturbWeight/ peturbDist user defined ( how much a weight will be increased/decreased)
	# make the peterbWeightProb also user defined (with)
	# Default, 90% probablility being uniformly peterbed, 10% being assigned a completely random value
	# maybe it can be: new value += random.uniform(low, high)
	# check line 299 for more info

	

		# Theres a preset chance that an inherited gene can become disabled if it is disabled in either parent
		# but then again, that's what the above loop does

		# genotypeToPhenotype constructs the network.
		# Maybe I can write a CUDA-friendly version of this function?
		#def genToPhen(self,genome)


# An ordered list of species is maintained.
# In each each generation, genomes are sequentially paced into species
# Echh existing species is representedby a random genome inside the species form the preivous generation}
# A given genome g in the current gfeneration is placed in the first species in which g is compatible with the prepresentative genome of that species
# This way, species do not overlap
# If g is not compativle with any existing speciess, a new species is created with g as its representative


# REPRODUCTION MECHANISM

# EXPLICIT FITNESS SHARING, Goldberg and Richardson, 1987.
# Organisms must share the fitness of their species
# The adjusted fitness fPrime[i] for organism [i] is calcuplted according to its distance delta from every other organism [j] in the population
# fPrime[i] = f[i]/[j for j in range(1,n): sum(sh(delta(i,j)))]
# The sharing function sh is set to 0 when the distance delta(i,j) is above the threshold deltaT
# otherwise, sh(delta(i,j) is set to 1 (Spears, 1995)

# if delta(i,j) > deltaT:
#	


# Converts genomes to actual machine instructions
# For the 2018 prosthetics challenge, this is the 'my_controller'
# inputs genomes and muscle exitations, outputs action


	# might need to make activationType a global parameter
	# The originial authors modified a sigmoid function so it would have greater steepness 
	# 1/(1+e^-4.9*x)
	def activation(self, node, actType, steepness):
		if actType == 'sig':
			#print(node)
			#activatedNode = 1/(1+math.exp(-steepness*node))
			activatedNode = 1/(1+np.exp(-steepness*node))
		elif actType == 'step':
			if node < 0:
				activatedNode = 0
			elif node >= 0:
				activatedNode = 1
		elif actType == 'tanh':
			activatedNode = np.tanh(node)
		elif actType == 'atan' or actType == 'arctan':
			activatedNode = np.atan(node)
		elif actType == 'ReLU' or actType == 'relu':
			if node < 0:
				activatedNode = 0
			elif node >= 0:
				activatedNode = node
		elif actType == 'PReLU' or actType == 'prelu' or actType == 'leak' or actType == 'leaky' or actType == 'leakyReLU' or actType == 'leakyRelu':
			if node < 0:
				activatedNode = steepness*node
			elif node >= 0:
				activatedNode = node
		elif actType == 'ELU' or actType == 'elu':
			if node < 0:
				activatedNode = steepness*(np.exp(node) - 1)
			elif node >= 0:
				activatedNode = node
		else:
			raise neatError('The activation type either be signmoid (sig), binary step (step), tanh, arctan (atan), Rectified Linear unit (relu), Parametrised/Leaky Rectified Linear Unit (prelu) or Exponential Linear Unit (elu)')
		return activatedNode
	# This 'fires' conenction genes (from one node to another)
	# also, each hidden node needs to be activated once all nodes have gone into it
	# the output of a hidden node with no more currinCons gets fed through an activation function
	# if a hidden or output node has no more currInCons, 
	# if self.currInCons(self.nodeNumList)
	def fireCon(self, nodes, index, nonFiredCons, activation, actType = 'sig', steepness = 1):
		""" I am ashamed at the inneficiency of this function.
		"""
		for i in range(0,len(self.nodeNumList)):
			# If a node number = the inCon of the firing gene
			if self.nodeNumList[i] == nonFiredCons[index][0]:
				# If a node number = the outCon of the firing gene
				for j in range(0, len(self.nodeNumList)):
					if self.nodeNumList[j] == nonFiredCons[index][1]:
						if activation == True:
							nodes[j] += self.activation(nodes[i] * nonFiredCons[index][2], actType, steepness)
						else:
							nodes[j] += nodes[i] * nonFiredCons[index][2]
		return nodes

	def isLoop(self, inpCon, conGenes = 'd'):
		if conGenes == 'd':
			conGenes = self.conGenes
		# a growing list of the nodes we've seen. prevents infinite loops
		observedNodes = []
		# a dynamic list of the current connections we're checking
		currCons = [inpCon]
		# the counter that loops through each current connection
		index = 0
		# the output boolean
		isLoopBool = False
		while len(currCons) > 0 and isLoopBool == False:
			# if the current connection goes to the in node of the input connection (inCon)
			if currCons[index][1] == inpCon[0]:
				isLoopBool = True
			# else if the target node has already been seen, remove that connection from the list of current connections
			elif currCons[index][1] in observedNodes:
				currCons.remove(currCons[index])
			# else, add the target node of that current Connection to the observed nodes list
			else:
				observedNodes.append(currCons[index][1])
				# append each connection that comes from the target node to the list of current connections
				# for each connection gene, if its inCon
				# if a connection gene comes from the outCon of the 
				#[currCons.append(conGenes[i]) for i in range(0, conGenes) if currCons[index][1] in self.currInCon(currCons[index][0], conGenes)]
				# add any connection that comes from that node
				[currCons.append(conGenes[i]) for i in range(0, len(conGenes)) if currCons[index][1] == conGenes[i][0]]
				# remove the current connection from the list of current connections
				currCons.remove(currCons[index])
				
			# increment the index by 1

		return isLoopBool

	def buildNetwork(self, inputs, actType= 'sig', steepness = 1):

		if len(self.sensor) != len(inputs):
			raise neatError("The number of sensors must match the number of input elements.")
		
		# setup
		nodeNumList = [i[0] for i in self.nodeGenes]
		nodes = []

		# for each node in the genome, set its value to 0
		[nodes.append(0) for i in self.nodeGenes]

		# a loop that writes the input values to the sensor values.
		# IT MIGHT LOOK STUPID, BUT THERES A REASON WHY
		# I did it this way, in case the order of nodes is messed up
		# i.e, [1,1],[2,2],[3,1]
		# setting up the sensor counter
		sensorIndex = 0
		# for every node
		for i in range(0,len(nodeNumList)):
			# I might change this line, kinda ineficient
			if nodeNumList[i] in self.sensor:
				nodes[i] = inputs[sensorIndex]
				sensorIndex += 1

		#print("testing the newest draft:")
		# copy the connection genes over to a list of non Fired Connections
		nonFiredCons = copy.copy(self.enabConGenes)
		#print(nonFiredCons)
		index1 = 0
		# There was a for 
		index2 = 0


		# new, experimental loop
		# all it does, is it fires any connections where the in node is a sensor,
		# once there are no more connections with an in node being a sensor, it moves on to the rest of the program
		while index1 < len(nonFiredCons) and nonFiredCons[index1][1] not in self.output:
			if nonFiredCons[index1][0] in self.sensor:
				#print('firing ' + str(nonFiredCons[index1]))
				nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
				nonFiredCons.remove(nonFiredCons[index1])
				index1 = 0
				#
				#index2 = 0
			else:
				index1 += 1

		# now that we've done that loop, lets move on to the rest of it
		index1=0

		# experimental
		noInConIndex = 0

		#boolean that  
		deadNetwork = False
		# While there are still non-fired connections about 46 lines under
		while len(nonFiredCons) > 0 and deadNetwork == False:
			##DEBUG STATEMENT
			#if index1 >= len(nonFiredCons):
			#	raise neatError("index1 is out of range; none of our rules can apply to this network.")
			# DEBUG STATEMENT: if there are no outputs in nonFiredCons outcons, raise an error 
			if len(set([i[1] for i in nonFiredCons]) - set(self.output)) == len(set([i[1] for i in nonFiredCons])):
				# I'll change this to some sort of boolean; 
				# this means that the network cannot be built, (and will be removed from the species?)
				# output = False???? or maybe skipBool = True
				# like, if a network can't be built, ignore it
				# maybe output [0,0,0,0,0,0,0,0,0,0] #but for the number of outputs in the network
				#raise neatError(" No output appears in the remaining connection's outcons.")
				deadNetwork = True
				outputs = []
				[outputs.append(0) for i in range(0,len(self.output))]

			# second, experimental loop
			# all it does is fire the connection genes with 0 inCons (or connection genes whose in node isn't in any of the other target nodes)
			while noInConIndex < len(nonFiredCons) and deadNetwork == False:
				if len(self.currDirInCons(nonFiredCons[noInConIndex][0])) == 0 or nonFiredCons[noInConIndex][0] not in [i[1] for i in nonFiredCons]:
					#print('firing ' + str(nonFiredCons[noInConIndex]))
					nodes = self.fireCon(nodes, noInConIndex, nonFiredCons, True, actType, steepness)
					nonFiredCons.remove(nonFiredCons[noInConIndex])
					noInConIndex = 0
					#
					#index2 = 0
				else:
					noInConIndex += 1
			# now that we're done with that loop, lets reset the counter and move on
			noInConIndex = 0

		# EXPERIMENTAL LOOP 27/07/2018

		# maybe its just a simple loop to see if there are any direct connections we should be concerned about

			selfLoopIndex = 0

			while selfLoopIndex < len(nonFiredCons):
				# if a node's only inCon is itself (i.e, self-looping), then fire it
				if len(self.currInCons(nonFiredCons[selfLoopIndex], nonFiredCons)) == 1 and nonFiredCons[selfLoopIndex][0] == nonFiredCons[selfLoopIndex][1]:
					#print('firing self loop connection ' + str(nonFiredCons[selfLoopIndex]))
					nodes = self.fireCon(nodes, selfLoopIndex, nonFiredCons, True, actType, steepness)
					nonFiredCons.remove(nonFiredCons[selfLoopIndex])
					selfLoopIndex = 0

				else:
					selfLoopIndex += 1


			# third experimental loop

			if index1< len(nonFiredCons):
			# i'll probably change this
				#while index1 <= len(nonFiredCons)-1:
				# if a gene has no inCons or if all inCons have been fired:
				#print(index1)
				#print(index2)
				#print(nonFiredCons)
				#print('looking at ', nonFiredCons[index1])
				#print('check if ', nonFiredCons[index1][0])
				#print('not in: ', [i[1] for i in nonFiredCons])
				if len(self.currDirInCons(nonFiredCons[index1][0])) == 0 or nonFiredCons[index1][0] not in [i[1] for i in nonFiredCons]:
					#print('firing ' + str(nonFiredCons[index1]))
					nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
					nonFiredCons.remove(nonFiredCons[index1])
					index1 = 0
				#
					index2 = 0
			# else if a gene loops to itself
				elif nonFiredCons[index1][0] == nonFiredCons[index1][1]:
				# and has no inCons (other than itself,)
				#if len(max([self.currInCons(nonFiredCons[i][0]) for i in range(0,len(nonFiredCons))])) <= 1:
				#if len(max([(set([nonFiredCons[i][0]]) - set(self.currInCons(nonFiredCons[i][0]))) for i in range(0,len(nonFiredCons))])) >= 1:

				# the nodes that connect to that gene
					# AKA, if there are any nonFiredConnections which also connect to the self-looping gene
					selfLoop = []
					[selfLoop.append(i) for i in range(0,len(nonFiredCons)) if nonFiredCons[i][1] == nonFiredCons[index1][1]]
				#if len(set(self.currInCons(nonFiredCons[index1][0]))-set([nonFiredCons[index1][1]]) - set([i[1] for i in nonFiredCons])) != len(nonFiredCons):
					if len(selfLoop) > 1:
					#print([set([nonFiredCons[index][0]]) - set(self.currInCons(nonFiredCons[index][0])) for i in range(0,len(nonFiredCons))])
						#print('self looping, but not yet')
						index1 += 1
					elif nonFiredCons[index1][0] in self.sensor:
						#print('firing but not activating')
						nodes = self.fireCon(nodes, index1, nonFiredCons, False)
						nonFiredCons.remove(nonFiredCons[index1])
						index1 = 0
						#
						index2 = 0
					else:
						#print('firing ' + str(nonFiredCons[index1]))
						nodes = self.fireCon(nodes, index1, nonFiredCons,True, actType, steepness)
						nonFiredCons.remove(nonFiredCons[index1])
						index1 = 0
					#
						index2 = 0

					#print('checkpoint 1')
				# if the 'in' node and the 'target' node are both outputs, fire the current connection
				elif nonFiredCons[index1][0] in self.output and nonFiredCons[index1][1] in self.output:
					#print('firing ' + str(nonFiredCons[index1]))
					nodes = self.fireCon(nodes, index1, nonFiredCons,True, actType, steepness)
					nonFiredCons.remove(nonFiredCons[index1])
					index1 = 0
					#
					index2 = 0


				# if only the 'target' node is an output, skip it for now
					#print('checkpoint2')
				elif nonFiredCons[index1][1] in self.output:
					#print('target node is an output: skipping for now')
					index1 += 1

					#print('checkpoint3')
			# if a connection gene's 'in' node has more outCons than the 'target' node has,
			# that connection gene is involved in a big loop
				elif len(self.currOutCons(nonFiredCons[index1][0], nonFiredCons)) > len(self.currOutCons(nonFiredCons[index1][1], nonFiredCons)) or (len(self.currOutCons(nonFiredCons[index1][0], nonFiredCons)) < len(self.currOutCons(nonFiredCons[index1][1], nonFiredCons)) and nonFiredCons[index1][0] == nonFiredCons[index1][1]):
					#print('gene is involved in a loop of 2 or more connection genes')
				
				# if index1 has checked each of the nonFiredCons
					if index1 == len(nonFiredCons):
					# look at the start again
						index1 = 0
					# increment index2
					#index2 += 1

				# if the target node only goes to the in node
				# aka, if the target node only has 1 outCon, and that outCon is the in node,
					if len(self.currOutCons(nonFiredCons[index1][1], nonFiredCons)) == 1 and nonFiredCons[index1][0] in self.currOutCons(nonFiredCons[index1][1], nonFiredCons):
					# fire it
						#print('firing a gene involved in a direct connection: ' + str(nonFiredCons[index1]))
						nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
						nonFiredCons.remove(nonFiredCons[index1])
						index1 = 0
						index2 = 0


				# not sure about this
				# if the 'in' node has fewer inCons than the 'target' node,
				# fire that connection gene
				# experimental checnge to outCons <= inCons
					elif len(self.currOutCons(nonFiredCons[index1][0], nonFiredCons)) > len(self.currInCons(nonFiredCons[index1][1], nonFiredCons)):
						#print('the in node has more outCons than the target has inCons.')
					# if that connection gene also has a self looping gene coming out of it:
					#if nonFiredCons[index1][0] in [i[1] for i in nonFiredCons]: 
						if nonFiredCons[index1][0] in self.currOutCons(nonFiredCons[index1][0], nonFiredCons):
					#and nonFiredCons[index1][1] in [i[1] for i in nonFiredCons]:
					#nonFiredCons[index1][1]: b`
							#print('not yet: self looping connection gene')
							#print('index1 currently is ', index1)
							index1 +=1

						#EXPERIMENTAL
						# else if the in node has less inCons than the target node has outCons, fire it
						elif len(self.currInCons(nonFiredCons[index1][0], nonFiredCons)) < len(self.currOutCons(nonFiredCons[index1][1], nonFiredCons)):
							#print('the in node has less inCons than the target node has outCons')
							#print('firing ' + str(nonFiredCons[index1]))
							nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
							nonFiredCons.remove(nonFiredCons[index1])
							index1 = 0
							index2 = 0

						elif nonFiredCons[index1][0] in self.output:
							#print('skipping big loop, experimental,cause in nodes is output')
							index1 += 1

						else:
							#print('skipping big loop')
							index1 += 1
				# else if the 'in' node and the 'target' node have the same number of inCons and outCons
				#elif len(self.currOutCons(nonFiredCons[index1][0], nonFiredCons)) == len(self.currInCons(nonFiredCons[index1][1], nonFiredCons)) and len(self.currOutCons(nonFiredCons[index1][1], nonFiredCons)) == len(self.currInCons(nonFiredCons[index1][0], nonFiredCons)):
					elif len(self.currOutCons(nonFiredCons[index1][0], nonFiredCons)) < len(self.currInCons(nonFiredCons[index1][1], nonFiredCons)):
					# if that connection gene also has a self looping gene coming out of it:
					#if nonFiredCons[index1][0] in [i[1] for i in nonFiredCons]: 
						if nonFiredCons[index1][0] in self.currOutCons(nonFiredCons[index1][0], nonFiredCons):
					#and nonFiredCons[index1][1] in [i[1] for i in nonFiredCons]:
					#nonFiredCons[index1][1]:
							#print('not yet: self looping connection gene')
							index1 +=1
						else:
							#print('firing big looped (same outCons and inCons): ' + str(nonFiredCons[index1]))
							nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
							nonFiredCons.remove(nonFiredCons[index1])
							index1 = 0
							index2 = 0
					else:
						index1 += 1
					#print('checkpoint4')

					# a very ugly but usable addition
					if index1 == len(nonFiredCons) and index2 == len(nonFiredCons):
						print("whoops")

			# else, loop the remaining connections for any looping/recurrent connections
				else:
					#index1 = 0
					#print('loop the remaining connections')
					index2 = 0
				# scan through the remaining nonFiredConnections from the beginning
				#index1 += 1
					fired = False
				#while index2 <= len(nonFiredCons)-1 and index1 != 0:
					while index2 <= len(nonFiredCons)-1 and index1 <= len(nonFiredCons)-1:
						if index2 == len(nonFiredCons):
						#index1 += 1
							index2 = 0
						#print('else,')

					# if a gene's outCon is in another inCon And that gene's inCon is in another outCon
					#if nonFiredCons[i][0] in [i[1] for i in nonFiredCons] and nonFiredCons[i][1] in [i[0] for i in nonFiredCons]:
					#print('checking the gene '+ str(nonFiredCons[i]) + ' against ' + str(nonFiredCons[index]))
					#Check for functionality here; it might be broken here
						if (nonFiredCons[index2][0] == nonFiredCons[index1][1] and nonFiredCons[index2][1] == nonFiredCons[index1][0]):
						# A series  of checks for these two looped connections:
						# If a connection gene connects directly to itself:

						# If a connection gene has less current output connections than another
							#print('a gene is directly recurrent')
							#print('gene ', nonFiredCons[index1], ' goes to ', nonFiredCons[index2])
						# maybe
						# make a function that tells you how to get from node 1 to node 2
						# which connections you need to take to go from node i to node j
						# if len(self.newfunction(nonFiredCons[index2][0])) > len(self.newFunction(nonFiredCons[index1][0]))
							#print('number of outcons from node ', nonFiredCons[index1][0], ', ', len(self.currOutCons(nonFiredCons[index1][0], nonFiredCons)))
						#print(set(self.currOutCons(nonFiredCons[index1][0])) - set([i[0] for i in self.enabConGenes]))
						
							#print('number of outcons from node ', nonFiredCons[index1][1], ', ', len(self.currOutCons(nonFiredCons[index1][1], nonFiredCons)))

							if len(self.currDirOutCons(nonFiredCons[index1][0], nonFiredCons)) == len(self.currDirOutCons(nonFiredCons[index1][1], nonFiredCons)):
								#print('the in node and target node have the same number of outCons')
							# if 4 is connected to 
							# else if the in node and target node ALSO have the number of inCons,
							# or if the only incon is the target node ( i.e, 1 to 2, 2 to 3, 3 to 2; 3's only in node)
							#if self.currDirOutCons(nonFiredCons[index1][0]) == self.currDirInCons(nonFiredCons[index1][1]))
								if len(self.currDirInCons(nonFiredCons[index1][0], nonFiredCons)) == len(self.currDirInCons(nonFiredCons[index1][1], nonFiredCons)) or self.currDirOutCons(nonFiredCons[index1][0]) == self.currDirInCons(nonFiredCons[index1][1]) or self.currDirOutCons(nonFiredCons[index2][0]) == self.currDirInCons(nonFiredCons[index2][1]):
									#print('the in node and target node also have the same number of inCons:')
									if nonFiredCons[index1][0] in self.output:
										fired = True
										#print('firing  ' + str(nonFiredCons[index1]))
										nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
										nonFiredCons.remove(nonFiredCons[index1])
										index1 = 0
										index2 = 0
									elif nonFiredCons[index2][0] in self.output:
										fired = True
										#print('firing ' + str(nonFiredCons[index2]))
										nodes = self.fireCon(nodes, index2, nonFiredCons, True, actType, steepness)
										nonFiredCons.remove(nonFiredCons[index2])
										index1 = 0
										index2 = 0
									# EXPERIMENTAL 27/07/2018
									else:
										# just fire both of them anyway
										#print('firing both genes')
										fired = True
										nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
										nonFiredCons.remove(nonFiredCons[index1])
										index1 = 0
										#print('jeff')
										#nodes = self.fireCon(nodes, index2, nonFiredCons, True, actType, steepness)
										#nonFiredCons.remove(nonFiredCons[index2])
										index2 = 0
							# if a gene's in node is a sensor, fire that gene

								elif len(self.currDirInCons(nonFiredCons[index2][0], nonFiredCons)) < len(self.currDirOutCons(nonFiredCons[index2][1], nonFiredCons)):
								# fire the larger connections to the smaller
									if nonFiredCons[index2][0] in self.sensor:
										fired = True
										#print('firing but not activating')
										# chagned activation to true
										nodes = self.fireCon(nodes, index2, nonFiredCons, True, actType, steepness)
									else:
										fired = True
										#print('firing ' + str(nonFiredCons[index2]))
										nodes = self.fireCon(nodes, index2, nonFiredCons, True, actType, steepness)
									nonFiredCons.remove(nonFiredCons[index2])
								#index1 += 1
									index1 = 0
									index2 = 0


							# EXPERIMENTAL!!!!
							elif nonFiredCons[index1][0] in self.output:
								#print('connection gene', nonFiredCons[index1], ' comes from an output')
								fired = True
								#print('firing but not activating')
								# 4/08/2018 changed activation to true
								nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
								nonFiredCons.remove(nonFiredCons[index1])
									#index1 += 1
								index1 = 0
									#index2 += 1
								index2 = 0

							# EXPERIMENTAL!!!!
							elif nonFiredCons[index2][0] in self.output:
								#print('connection gene', nonFiredCons[index2], ' comes from an output')
								fired = True
								#print('firing but not activating')
								# changed activation to true
								nodes = self.fireCon(nodes, index2, nonFiredCons, True, actType, steepness)
								nonFiredCons.remove(nonFiredCons[index2])
									#index1 += 1
								index1 = 0
									#index2 += 1
								index2 = 0

							# if both looped connections have the same number of outward connections,
							elif len(self.currDirOutCons(nonFiredCons[index2][0], nonFiredCons)) == len(self.currDirOutCons(nonFiredCons[index1][0], nonFiredCons)):
							#if len(self.currInCons(nonFiredCons[index2][1], nonFiredCons)) > len(self.currInCons(nonFiredCons[index1][1], nonFiredCons)):
							#	index1 += 1
							#elif len(self.currInCons(nonFiredCons[index2][1], nonFiredCons)) < len(self.currInCons(nonFiredCons[index1][1], nonFiredCons)):
							#	index2 += 1
							# check to see if node i is an output
								if nonFiredCons[index2][0] in self.output:
								# fire i to j
									fired = True
									#print('firing ' + str(nonFiredCons[index2]))
									nodes = self.fireCon(nodes, index2, nonFiredCons, True, actType, steepness)
									nonFiredCons.remove(nonFiredCons[index2])
									#index1 += 1
									index1 = 0
									index2 = 0
							# check to see if node j is an output
								elif nonFiredCons[index1][0] in self.output:
									fired = True
									#print('firing ' + str(nonFiredCons[index1]))
									nodes = self.fireCon(nodes, index1, nonFiredCons, True, actType, steepness)
									nonFiredCons.remove(nonFiredCons[index1])
								#index1 += 1
									index1 = 0
								#index2 += 1
									index2 = 0
							# if neither looped connection is an output, raise an error
								else:
								#raise neatError('neither connected node is an output. (impossible to build)')
								#index1 += 1
									index2 += 1
					# if the inCon of a gene is an output, fire it
						elif nonFiredCons[index2][0] in self.output:
							fired = True
							#print('firing ' + str(nonFiredCons[index2]))
							nodes = self.fireCon(nodes, index2, nonFiredCons, True, actType, steepness)
							nonFiredCons.remove(nonFiredCons[index2])
						#???index1 += 1
							index1 = 0
							index2 = 0
					# if the outCon of a gene is an output, skip it
					# might remove the line
					#elif nonFiredCons[index2][1] in self.output:
					#	index2 += 1
						index2 += 1

						#index1 +=1

				# if a gene has been fired, set the counter back to 0, else, increment it
					if fired == False:
						index1 += 1
					else:
						index1 = 0

					#31/07/2018
					# else if a connection has 2 or more outCons
					#print('tried the main while loop, trying experimental')
					#print(nonFiredCons)
					# i believe this stops us from checking connections we've already seen? I'm too tired, i dont remember
					index2 = 0
					# the big loop boolean lets us go back to the original loop once I fire a connection
					bigLoopBool = True
					while index2 < len(nonFiredCons) and bigLoopBool == True and deadNetwork == False:
						#DEBUG STATEMENT
						#if index1 >= len(nonFiredCons):
						if index1 > len(nonFiredCons):
							deadNetwork = True
							outputs = []
							[outputs.append(0) for i in range(0,len(self.output))]
								#raise neatError("index1 is out of range; none of our rules can apply to this network.")
						#print('index2 is ', index2)
						elif len(self.currOutCons(nonFiredCons[index2][0],nonFiredCons)) >= 2:
							#print('original gene:', nonFiredCons[index1])
							#print('trying experimental one; connection gene:', nonFiredCons[index2])
							fireNonLoops = False
							loopIndex = 0
							# for each of the outCons
							#while loopIndex < len(self.currOutCons(nonFiredCons[index1][0],nonFiredCons)):
							while loopIndex < len(nonFiredCons) and fireNonLoops == False:
								# if a connection comes from the desired in node AND is a nonLoop connection
								if nonFiredCons[loopIndex][0] == nonFiredCons[index2][0] and self.isLoop(nonFiredCons[loopIndex],nonFiredCons) == False:
								#	print('jeff1')
									fireNonLoops = True
								#	print('this will end the current loop')
								loopIndex += 1


							if fireNonLoops == True:
								#print('jeff2')
								# reset loop index
								loopIndex = 0
								while loopIndex < len(nonFiredCons):
									# if a connection gene comes from the desired in node AND is a looping connection,
									if nonFiredCons[loopIndex][0] == nonFiredCons[index2][0] and self.isLoop(nonFiredCons[loopIndex],nonFiredCons) == True:
										#fire that connection
										# If this fires everything
										# I want to find a way to 'break' the loops without 'breaking'
										#print('firing the isLoop ' + str(nonFiredCons[loopIndex]))
										nodes = self.fireCon(nodes, loopIndex, nonFiredCons, True, actType, steepness)
										nonFiredCons.remove(nonFiredCons[loopIndex])
										loopIndex = 0
										index1 = 0
										index2 = 0
										# get out of this secondary loop, go back to the original one
										bigLoopBool = False
									else:
										loopIndex += 1
									#print('jeff3')
					
							else:
								index1+=1
						#print('skip')
						index2+=1
				index2+=1

			
		#print(nonFiredCons)
		# outputs is a list of the output nodes you want to return
		# it's based on the structure of the 'nodes' list
		outputs = []
		for i in range(0,len(self.nodeNumList)):
			# If a node number is an output
			if self.nodeNumList[i] in self.output:
				# add the value of that node to the outputsList
				outputs.append(nodes[i])

		return outputs


class mate():
	def compDist(genome1, genome2,c1=1.0,c2=1.0,c3=1.0):
		# Wbar is the weight differences of the matcihng genes, including disabled genes
		#delta = c1*E/N + c2D/N + c3*Wbar

		# N reprents the number of genes in the larger genome
		# N = len(max(genome1.conCount, genome2.conCount))
		# if N < 20:
			# N = 1
		#If both genomes have less than 20 genes, set N to 1

		# NOTE: This is a copy/paste of the code below
		# I'll probably change this garbled mess

		###############################

		# just to make it a bit easier to work with all these transformations
		#match = []
		# Sets up a list structure; index 0 is for genome1, index 1 is for genome2
		sharedGenes = [[],[]]
		#sharedGenes = []
		# Sets up another list structure, combining genome 1 and genome 2
		genePool = [genome1.conGenes,genome2.conGenes]

		# sets up the output; a new genome with those connection genes
		#output = []

		# The innovation numbers for genome 1, as a temporary variable
		innovNum1 = [i[4] for i in genome1.conGenes]
		innovNum2 = [i[4] for i in genome2.conGenes]
		#bothInnovs = list(set(innovNum1) | set(innovNum2))

		# for some reason, I have to make the 'and' an ampersand for it to work
		#match.append(list(set(innovNum1) & set(innovNum2)))
		# ADDITION (27/08/18)
		match = list(set(innovNum2) & set(innovNum1))
		#print()


		nonMatch = []

		nonMatch.append(list(set(innovNum1) - set(innovNum2)))
		# The values in genome 2 that arent in genome 1
		nonMatch.append(list(set(innovNum2) - set(innovNum1)))

		
		######## Find the genomes which 4th element are in Match 
		# There must be a better way to do this!!!!!
		#for i in range(0, genome1.conCount):
		# for i in range(0, len(genePool[0])):
		# 	#if (innovNum1[i] in match[0]) and (innovNum2[i] in match[0]):
		# 	#if innovNum1[i] not in nonMatch[0] and innovNum1[i] not in nonMatch[1]:
		# 	if innovNum1[i] in match:
		# 		# # check that we haven't already added that gene into shared genes 
		# 		# for j in range(0,len(sharedGenes[0])):
		# 		# 	if innovNum1[i] not sharedGenes[j]
		# 		sharedGenes[0].append(i)
		# 		#sharedGenes.append(i)

		# # # for each connection gene in genome 2,
		# for i in range(0, len(genePool[1])):
		# # 	# if a connection gene is in the matching set
		# 	#if innovNum2[i] not in nonMatch[0] and innovNum2[i] not in nonMatch[1]:
		#  	if innovNum2[i] in match:
		# # 		# append genome 2's corresponding index
		#  		sharedGenes[1].append(i)

		# a better version (28/08/18)
		for i in range(0,len(innovNum1)):
			innovNumIndex = 0
			common = False
			# increment through the other genome to find matching innovation numbers
			while innovNumIndex < len(innovNum2) and common == False:
				# if we find a matching pair
				if innovNum1[i] == innovNum2[innovNumIndex]:
					# add the indexes (the counter for where to find them) to the sharedGenes list
					sharedGenes[0].append(i)
					sharedGenes[1].append(innovNumIndex)
					common = True
				else:
					innovNumIndex += 1


		# print('error???')
		# print(genePool)
		# print('genome1 concount:')
		# print(genome1.conCount)
		# print('genome2 concount: ')
		# print(genome2.conCount)
		# print('sharedGenes:')
		# print(sharedGenes)

		# nonMatch is a list of all the non matching genes
		nonMatch = []
		disjoint = []
		excess = []

		# The values in genome 1 that arent in genome 2
		# [i[4] for i in genome1] a list of all the enabled values in a genome
		# maybe make temp1 = [i[4] for i in genome1.conGenes]
		# and temp 2 = [i[4] for i in genome2.conGenes]

		nonMatch.append(list(set(innovNum1) - set(innovNum2)))
		# The values in genome 2 that arent in genome 1
		nonMatch.append(list(set(innovNum2) - set(innovNum1)))

		#nonMatch -> [genes in genome1 not in genome2, genes in genome2 not in genome 1]

		# If there no non matching genes (all genes match for both)
		#if len(nonMatch[0]) == 0 and len(nonMatch[1]) == 0:
			#print("Both genes match. Will randomly choose genes to crossover.")
			# Randomly choose each of the genes to breed
		# else if the genomes have different fitness values,
		###elif genome1.fitness != genome2.fitness:
			# for each of the non matching genes in genome 1 that aren't in genome 2,
		for i in range(len(nonMatch[0])):

				# if nonMatch[i] is not within the range of the other genome 
			if nonMatch[0][i] < min(innovNum2) or nonMatch[0][i] > max(innovNum2):
					# It is an excess gene
				excess.append(nonMatch[0][i])
			else:
					# It is a disjoint gene
				disjoint.append(nonMatch[0][i])

			# for each of the non matching genes in genome 2 that aren't in genome 1,
		for i in range(len(nonMatch[1])):

				# if nonMatch[i] is not within the range of the other genome 
			if nonMatch[1][i] < min(innovNum1) or nonMatch[1][i] > max(innovNum1):
					# It is an excess gene
				excess.append(nonMatch[1][i])
			else:
					# It is a disjoint gene
				disjoint.append(nonMatch[1][i])

		# print('troubleshooting!')
		# print('Excess genes: ', excess)
		# print('Disjoint genes: ', disjoint)


		# A stupid addition (27/08/18)
		# To get rid of a problem of genes repeating themselves in a genome, I have this temporary fix
		# if there's more than one of the same gene, it will randomly remove all but one.
		# sharedGenesIndex = 0
		# while sharedGenesIndex < len(sharedGenes[0]):
		# 	# if an index from sharedGenes[0] isn't in sharedGenes[1], remove it
		# 	if sharedGenes[0][sharedGenesIndex] not in sharedGenes[1]:
		# 		# remove it from the sharedGenes list
		# 		sharedGenes[0].remove(sharedGenes[0][sharedGenesIndex])
		# 	else:
		# 		sharedGenesIndex += 1

		# sharedGenesIndex = 0
		# while sharedGenesIndex < len(sharedGenes[1]):
		# 	# if an index from sharedGenes[1] isn't in sharedGenes[0], remove it
		# 	if sharedGenes[1][sharedGenesIndex] not in sharedGenes[0]:
		# 		# remove it from the sharedGenes list
		# 		sharedGenes[1].remove(sharedGenes[1][sharedGenesIndex])
		# 	else:
		# 		sharedGenesIndex += 1

		# The average weight differences of the matching genes, including disabled genes
		#wBar = ([i for i in sharedGenes[0]] in genePool[0]] - [i[3] for i in sharedGenes[1]])/len(sharedGenes[0])
		
		# diffs is a list of all the weight differences of matching genomes
		diffs = []
		for i in range(len(sharedGenes[0])):
			diffs.append(genePool[0][sharedGenes[0][i]][2] - genePool[1][sharedGenes[1][i]][2])
			#print(genePool[0][sharedGenes[0][i]] - genePool[1][sharedGenes[1][i]])
		if len(diffs) == 0:
			wBar = 0
		else:
			wBar = sum(diffs)/len(diffs)
		if max(genome1.conCount,genome2.conCount) < 20:
			N = 1
		else:
			N = max(genome1.conCount,genome2.conCount)
		delta = c1*len(excess)/N + c2*len(disjoint)/N + c3*wBar
		print('The compatibility distance between the two genomes is ', delta)
		return delta


	def crossBreed(genome1, genome2, prob = 0.75):
		print('genome1 is ', genome1, '. It is from species ', genome1.speciesNum, '. It is of generation ', genome1.generationNum)
		print('its fitness is ', genome1.fitness)
		print('genome2 is ', genome2, '. It is from species ', genome1.speciesNum, '. It is of generation ', genome2.generationNum)
		print('its fitness is ', genome2.fitness)
		# Write the variable 'prob' to the instance variable 'self.prob'
		if not(type(prob) == float or type(prob) == int):
			raise neatError("Prob must be a number between 0 and 1.")
		if prob > 1 or prob < 0:
			raise neatError("Prob must be a number between 0 and 1. ")
		# maybe indent it???
		#if len(genome1.fitness) == 0 or len(genome2.fitness) == 0:
		if genome1.fitness == 'empty' or genome2.fitness == 'empty':
			raise neatError("Both genes must have a fitness score to be crossbreed.")
		##### IMPORTANT
		# When crossing over, the genes with the same innovation numbers are lined up
		# and are randomly chosen for the offspring genome.
		# Genes that do not match are inherited from the more fit parent,
		# or if they are equally fit, from both paarents randomly.

		# Disabled genes have a 25% chance of being reenabled during crossover

		# just to make it a bit easier to work with all these transformations
		#match = []
		# Sets up a list structure; index 0 is for genome1, index 1 is for genome2
		sharedGenes = [[],[]]
		# Sets up another list structure, combining genome 1 and genome 2
		genePool = [genome1.conGenes,genome2.conGenes]
		# arbitrary; obtains the node information from either parent
		# CHANGE THIS LINE (29/08/18)
		childNodes = genome1.nodeGenes

		#NOTE: finally changing this line (14/09/18)
		# for every node in genome2
		for i in genome2.nodeGenes:
		# if a given node in genome2 isn't in genome1,
			if i not in childNodes:
		# add that node from genome2 to the childNodes
				childNodes.append(i)


		# sets up the output; a new genome with those connection genes
		childCons = []

		# The innovation numbers for genome 1, as a temporary variable
		innovNum1 = [i[4] for i in genome1.conGenes]
		innovNum2 = [i[4] for i in genome2.conGenes]
		bothInnovs = list(set(innovNum1) | set(innovNum2))

		# for some reason, I have to make the 'and' an ampersand for it to work
		#match.append(list(set(innovNum1) & set(innovNum2)))
		match = list(set(innovNum1) & set(innovNum2))
		######## Find the genomes which 4th element are in Match 
		# There must be a better way to do this!!!!!
		#for i in range(0,genome1.conCount):
		# for i in range(0,len(genePool[0])):
		# 	if innovNum1[i] in match:
		# 		sharedGenes[0].append(i)
		# 		#sharedGenes.append(i)

		# # for each connection gene in genome 2,
		# #for i in range(0,genome2.conCount):
		# for i in range(0,len(genePool[1])):
		# 	# if a connection gene is in the matching set
		# 	if innovNum2[i] in match:
		# 		# append genome 2's corresponding index
		# 		sharedGenes[1].append(i)


		# a better version (28/08/18)
		for i in range(0,len(innovNum1)):
			innovNumIndex = 0
			common = False
			# increment through the other genome to find matching innovation numbers
			while innovNumIndex < len(innovNum2) and common == False:
				# if we find a matching pair
				if innovNum1[i] == innovNum2[innovNumIndex]:
					# add the indexes (the counter for where to find them) to the sharedGenes list
					sharedGenes[0].append(i)
					sharedGenes[1].append(innovNumIndex)
					common = True
				else:
					innovNumIndex += 1

		print('the shared Genes are: ', sharedGenes)
		#if len(sharedGenes[0]) != len(sharedGenes[1]):
		#	raise neatError("shared Genes must be the same length")
		# However, today, I'm going to do a looping method.
		# That way, I can also extract both the shared innovation numbers,
		# but also what the element count for both genomes are.

		# nonMatch is a list of all the non matching genes
		nonMatch = []
		disjoint = []
		excess = []

		# The values in genome 1 that arent in genome 2
		# [i[4] for i in genome1] a list of all the enabled values in a genome
		# maybe make temp1 = [i[4] for i in genome1.conGenes]
		# and temp 2 = [i[4] for i in genome2.conGenes]

		nonMatch.append(list(set(innovNum1) - set(innovNum2)))
		# The values in genome 2 that arent in genome 1
		nonMatch.append(list(set(innovNum2) - set(innovNum1)))

		#nonMatch -> [genes in genome1 not in genome2, genes in genome2 not in genome 1]

		# If there no non matching genes (all genes match for both)
		if len(nonMatch[0]) == 0 and len(nonMatch[1]) == 0:
			print("Both genes match. Will randomly choose genes to crossover.")
			# Randomly choose each of the genes to breed
		# else if the genomes have different fitness values,
		###elif genome1.fitness != genome2.fitness:
			# for each of the non matching genes in genome 1 that aren't in genome 2,
		for i in range(len(nonMatch[0])):

				# if nonMatch[i] is not within the range of the other genome 
			if nonMatch[0][i] < min(innovNum2) or nonMatch[0][i] > max(innovNum2):
					# It is an excess gene
				excess.append(nonMatch[0][i])
			else:
					# It is a disjoint gene
				disjoint.append(nonMatch[0][i])

			# for each of the non matching genes in genome 2 that aren't in genome 1,
		for i in range(len(nonMatch[1])):

				# if nonMatch[i] is not within the range of the other genome 
			if nonMatch[1][i] < min(innovNum1) or nonMatch[1][i] > max(innovNum1):
					# It is an excess gene
				excess.append(nonMatch[1][i])
			else:
					# It is a disjoint gene
				disjoint.append(nonMatch[1][i])


		# A stupid addition (27/08/18)
		# To get rid of a problem of genes repeating themselves in a genome, I have this temporary fix
		# if there's more than one of the same gene, it will randomly remove all but one.
		# sharedGenesIndex = 0
		# while sharedGenesIndex < len(sharedGenes[0]):
		# 	# if an index from sharedGenes[0] isn't in sharedGenes[1], remove it from the parent genome?
		# 	if sharedGenes[0][sharedGenesIndex] not in sharedGenes[1]:
		# 		# remove it from that parent genome
		# 		genePool[0].remove(genePool[0][sharedGenes[0][sharedGenesIndex]])
		# 		# remove it from the sharedGenes list
		# 		sharedGenes[0].remove(sharedGenes[0][sharedGenesIndex])
		# 	else:
		# 		sharedGenesIndex += 1

		# sharedGenesIndex = 0
		# while sharedGenesIndex < len(sharedGenes[1]):
		# 	# if an index from sharedGenes[1] isn't in sharedGenes[0], remove it from the parent genome?
		# 	if sharedGenes[1][sharedGenesIndex] not in sharedGenes[0]:
		# 		# remove it from that parent genome
		# 		genePool[1].remove(genePool[1][sharedGenes[1][sharedGenesIndex]])
		# 		# remove it from the sharedGenes list
		# 		sharedGenes[1].remove(sharedGenes[1][sharedGenesIndex])
		# 	else:
		# 		sharedGenesIndex += 1

		print('the updated shared Genes are:', sharedGenes)

		# print("The unique innovation numbers in the two genomes are: ")
		# print(max(len(genePool[0]), len(genePool[1])))
		# print("The matching innovation numbers are: ")
		# print(match)
		# print("The non-matching innovation numbers are: ")
		# print(nonMatch)
		# print("The shared innovation numbers are: ")
		# print(sharedGenes)
		# print("The excess innovation numbers are: ")
		# print(excess)
		# print("The disjoint innovation numbers are: ")
		# print(disjoint)
		# #for i in range(genome1):
		# print("Genome 1: ")
		# print(genePool[0])
		# print("Genome 2:")
		# print(genePool[1])
		# print("Output Genes: ")
		#print([i[4] for i in genePool[1]])

			#print(list(set(innovNum1) | set(innovNum2)))
			#print([i[4] for i in genePool[0]])|set([i[4] for i in genePool[1]])
			#for i in range(len(list(set([i[4] for i in genePool[0]])|set([i[4] for i in genePool[1]])))):
			####for i in range(len(list(set(innovNum1) | set(innovNum2)))):

		# will likely change this
		matchIndex = 0
		# A count of which nonMatching index we're at for genomes 1 and 2
		nonMatchIndex1 = 0
		nonMatchIndex2 = 0

		if genome1.fitness != genome2.fitness:
			for i in range(0,len(bothInnovs)):

				if bothInnovs[i] in nonMatch[0] and genome1.fitness > genome2.fitness:
					childCons.append(genePool[0][nonMatchIndex1])
					nonMatchIndex1 += 1
				# prevents the gene from being added twice
				elif bothInnovs[i] in nonMatch[1] and genome1.fitness < genome2.fitness:
					childCons.append(genePool[1][nonMatchIndex2])
					nonMatchIndex2 += 1

				# Note that this if statement is the same for if both genes have the same fitness
				# As seen below

				if bothInnovs[i] in match:
				# I'd like to  change this
					randTemp = random.choice([0,1])
					childCons.append(genePool[randTemp][sharedGenes[randTemp][matchIndex]])
					matchIndex += 1
				# We're now looking at the next innovation number, so we need to increment both
					nonMatchIndex1 += 1
					nonMatchIndex2 += 1

			# else if the genomes have the same fitness,
		elif genome1.fitness == genome2.fitness:
			# for each of the non matching genes in genome 1 that aren't in genome 2,
			for i in range(0,len(bothInnovs)):
				randTemp = random.choice([0,1])
				# Genes are randomly selected
				if bothInnovs[i] in nonMatch[0]:
					if randTemp == 1:
						childCons.append(genePool[0][nonMatchIndex1])
					nonMatchIndex1 += 1
				# prevents the gene from being added twice
				elif bothInnovs[i] in nonMatch[1]:
					if randTemp == 1:
						childCons.append(genePool[1][nonMatchIndex2])
					nonMatchIndex2 += 1

				if bothInnovs[i] in match:
				# I'd like to  change this
					randTemp = random.choice([0,1])
					childCons.append(genePool[randTemp][sharedGenes[randTemp][matchIndex]])
					matchIndex += 1
				# We're now looking at the next innovation number, so we need to increment both
					nonMatchIndex1 += 1
					nonMatchIndex2 += 1

			# randomly choose if *any* genes from both genomes appear in the output
			# That means disjoint and excess genes have a *50/50* chance of appearing in the output, too!
		# return output


		# for each gene in the output,
		for i in range(len(childCons)):
			# If a genome is disabled
			if childCons[i][3] == 0:
				# it has a prob of being re-enabled.
				# The preset probability is 75%
				# (The [0] at the end is because random.choices outputs a list)
				childCons[i][3] = random.choices([0,1], weights = [prob, 1-prob])[0]

		# weird formatting, but its cause it's a tuple, so I have to convert it to a list
		# NOTE:
		# 24/08/2018
		# it just outputs the node genes and connection genes
		# I want to output a fully fledged genome; with all the other components
		#return (childNodes,childCons)[0]
		print('parents are:')
		print(genePool)
		print('child of parents is:')
		print([childCons])


		# NOTE: I'm STILL getting repeating connection genes! (like, [1,173],[1,173]) (2/10/18)
		# This loop uses python's in-built set library; so the comparisons are super fast
		# I'm trying to avoid making this O(N^2)

		index1 = 0
		# a secondary 'cursor' that searches for copies of a connection gene
		while index1 < len(childCons):
			index2 = 0
			# if we were to remove the current conenction gene (putting it in a temp),
			tempConnection = childCons[index1]
			childCons.remove(tempConnection)
			# if the gene STILL appears in the conGenes list,
			if tempConnection[4] in [i[4] for i in childCons]:
				while index2 < len(childCons):
					# if there is another connection with the same innovation number,
					if childCons[index2][4] == tempConnection[4]:
					# remove that cloned connection
						childCons.remove(childCons[index2])
					else:
						index2 += 1
			# add our original connection back into the list
			childCons.append(tempConnection)
			# increment to the next unique connection in childCons
			index1 += 1

		return [childNodes,childCons]
	#meme
class species():

	# AN IDEA FORM A WORKSHOP:
	# 3 LISTS WITH STRINGS: 
	#country =['australia','usa']
	#states = ['WA', 'washington','ACT']
	# location = ['white house','parliament house', 'bell tower']

	#def linkThemAll():
	# First, we'll link all the locations to states
	# for each state entry;
		# for each state location: 
			# if the state location j is a part of state i, link the two
	#for i in range()
	# Then, we'll link all the states (which also have the locations linked) to the countries

	#[[species1],[species2],[species3]]
	#[[generation1,gener2,gener3],[gener1,gener2,gener3],[gener1,gener2,gener3]]
	#[[genome1,genome5,genome8],[genome2,genome6],[genome3, genome7]]
	#[[]]

	# genome is created, it's given a generation number attribute
	# once it's done the simulation and given a fitness value, its sorted into a species
	# Once in a species, it's put in its corresponding generation list (or that list is made in that species)

	# The species list and generation lists act as indexes:
	# The species list tells the program which
	def __init__(self, speciesList = 'd', generationList = 'd', genomeList = 'd', generationNum = 0, speciesNum = 0):
		if speciesList == 'd':
			self.speciesList = []
		else:
			self.speciesList = speciesList

		if generationList == 'd':
			self.generationList = []
		else:
			self.generationList = generationList

		if genomeList == 'd':
			self.genomeList = []
		else:
			self.genomeList = genomeList
		#dfgdf

	def display(self):
		print('\n')
		print('Displaying the 3 lists for species: ')
		print("Species List: " + str(self.speciesList))
		print("Number of Species: ", len(self.speciesList))
		generationLengths = []
		[generationLengths.append(len(self.speciesList[i])) for i in range(0,len(self.speciesList))]
		print("Number of generations in each species: ", generationLengths)
		speciesLengths = []
		# a list of the lengths of each species
		[speciesLengths.append(self.genomeCount(i)) for i in range(0,len(self.speciesList))]
		#print("Size of each species (in genomes): ", speciesLengths#[0]?)
		print("Size of each species (in genomes): ", speciesLengths)
		print("Generation List: " + str(self.generationList))
		print("Genome List: " + str(self.genomeList))

	def place(self,inputGenome, deltaT, c1=1.0, c2=1.0, c3=1.0):
		if type(inputGenome.fitness) == str:
			raise neatError('genomes can only be placed if they have been simulated/have a fitness value')
		# deltaT is the compatability threshold
		# for each species,
		#print('input genome: ', inputGenome)
		#print('input genome is of type: ', type(inputGenome))
		index = 0
		inserted = False
		# if there are no species yet
		if len(self.speciesList) == 0:
			# add a new species
			self.speciesList.append([])
			# add a generation in that species
			self.speciesList[len(self.speciesList)-1].append([])
			#print('our species now looks like: ', self.speciesList)
			# add our genome in that species' generation
			# note the 0; referring to the 1 and only generation in that species
			self.speciesList[len(self.speciesList)-1][0].append(inputGenome)
			inputGenome.speciesNum = len(self.speciesList)-1
			#print('after adding in ', inputGenome, ': ', self.speciesList)
		else:
			while index < len(self.speciesList) and inserted  == False:
				# if the genome is compatible with the latest generation of the ith species,
				#print('species list (line 1492) is met: ', self.speciesList)
				#print('genomes we are choosing from: ', self.speciesList[index][len(self.speciesList[index])-1])
				# Note: considering changing the  -1 to -2. This is cause I got an error 'can't choose from an empty sequence'. (8/09/18)
				if mate.compDist(inputGenome,random.choice(self.speciesList[index][len(self.speciesList[index])-1]), c1, c2, c3) <= deltaT:
					# if our genome is from the same generation as the 'latest' ith species genomes
					# THIS IS THE MOST IMPORTANT LINE IN THE FUNCTION/CLASS!!!!
					# note: changing -1 to -2, as it pools the simulated genomes into the same group as their parents, causing exponential problems. (13/09/18)
					if inputGenome.generationNum == self.speciesList[index][len(self.speciesList[index])-2][0].generationNum:
						# add our genome to the latest generation of the ith species
						self.speciesList[index][len(self.speciesList[index])-1].append(inputGenome)
						# Nice bookkeeping; we can see which species a genome belongs to
						inputGenome.speciesNum = index
						inserted = True
					else:
						# add a new generation to the ith species
						self.speciesList[index].append([])
						# add our genome to the latest generation in the ith species
						# Nice bookkeeping; we can see which species a genome belongs to
						# python starts at 0, humans read starting at 1
						inputGenome.speciesNum = index
						self.speciesList[index][len(self.speciesList[index])-1].append(inputGenome)
						inserted = True
				else:
					index += 1
			if inserted == False:
				# add a new species
				self.speciesList.append([])
				# add a generation in that species
				self.speciesList[len(self.speciesList)-1].append([])
				# add our genome in that species' generation
				# note the 0; referring to the 1 and only generation in that species
				#print('adding in a genome! ', inputGenome)
				# Nice bookkeeping; we can see which species a genome belongs to
				inputGenome.speciesNum = len(self.speciesList)-1
				self.speciesList[len(self.speciesList)-1][0].append(inputGenome)
				inserted = True

	# if the maximum fitness of a species did not improve in 15 generations, 
	# the networks in that species aren't allowed to reproduce (i'll delete them)
	def cull(self,inputGenome, stagSpeciesThreshold = 15):
		index = 0
		while index < len(self.speciesList):
			# Note the -1. This is because humans count from 1 onwards, but python counts the first generation as 0. (2/09/18)
			# if the latest genomes in the ith species haven't been improved in n generations
			# AKA. if the latest genomes from the ith species are more than 15 generations fewer than the current generation Number 
			if self.speciesList[index][len(self.speciesList[index])-2][0].generationNum + stagSpeciesThreshold < inputGenome.generationNum:
				# remove that species; it's extinct
				self.speciesList.remove(self.speciesList[index])
				print('removing species number ', index)
			else:
				# go to the next species
				index += 1


	# def champGenome(self, speciesNum, champGenomeThreshold=5):
	# 	#for each generation in the ith species,
	# 	for i in range(0,len(speciesNum)):
	# 		# if a species has more than 5 networks in it,
	# 		if self.genomeCount(self.speciesList[speciesNum]) > champGenomeThreshold:
	# 			# add a new generation (list) to that species
	# 			self.speciesList[speciesNum].append([])
	# 			# copy the best performing genome to the next generation of that species (unchanged)
	# 			#print('the latest generation in that species is:')
	# 			#print(self.speciesList[i][len(self.speciesList[i])-1])
	# 			self.speciesList[i][len(self.speciesList[i])-1].append(copy.copy(self.largestFitness(i)))

	# returns the number of networks in the ith species
	# (30/08/18) I am adding an argument to genomeCount that allows us to ignore a set number of generations from a species ( default 0 )
	# (some species will have genomes that havent been simulated yet)
	def genomeCount(self, speciesIndex, ignoreGeneration = 0):
		# this is a counter for all the genomes in a species. starts out at 0, obviously
		totalNum = 0
		# counts up the number of genomes in that species
		# for every generation in that species
		# 
		for i in range(0,len(self.speciesList[speciesIndex]) - ignoreGeneration):
			# add the number of genomes in that generation in that species to the totalNum
			totalNum += len(self.speciesList[speciesIndex][i])

		return totalNum
	# returns the genome with the largest fitness in that species
	# Note: added in ignoreGeneration (if we only want to find the best genome from everything but the last n generations of a species). (2/09/18)
	def largestFitness(self, speciesIndex, ignoreGeneration = 0):
		# set up the 'champion' genome as some arbitrary thing.
		champion = self.speciesList[speciesIndex][0][0]
		# for each generation of the ith species
		#print('for debugging:')
		#print('the genomes in the species are:', self.speciesList[speciesIndex])
		# note the -1; cause we have a new species, we're not going to look in a blank list
		for i in range(0,len(self.speciesList[speciesIndex])-1):
			# for each genome in that ith species
			#print('we are in generation ',i, ' of species ', speciesIndex)
			for j in range(0,len(self.speciesList[speciesIndex][i])-ignoreGeneration):
				# if that genome's fitness value is higher
				print('current fitness of genome from species ', speciesIndex, ', generation ', i, self.speciesList[speciesIndex][i][j].fitness)
				print('the best genome we have currently seen in species ', speciesIndex, champion.fitness)
				
				# If a genome hasn't been simulated, it's default fitness is 'empty'
				if type(self.speciesList[speciesIndex][i][j].fitness) != str:
					if self.speciesList[speciesIndex][i][j].fitness > champion.fitness:
						champion = self.speciesList[speciesIndex][i][j]
		return champion

	def fitnessAdjust(self, speciesIndex):
		#self.genomeCount(speciesIndex)?????
		totalNum = genomeCount(speciesIndex)

		# now, for each genome in the species, we'll divide a given genome's fitness value by the total number of genomes in that species
		for i in range(0,len(self.speciesList[speciesIndex])):
			for j in range(0,len(self.speciesList[speciesIndex][i])):
				# error checking
				if self.speciesList[speciesIndex][i][j].fitness == 'empty':
					self.speciesList[speciesIndex][i][j].fitness = 0

				# resets each genome's adjusted fitness to 0
				self.speciesList[speciesIndex][i][j].adjFitness = 0
				# the adjusted fitness is the genome's fitness divided by the unmber of genomes in the species
				self.speciesList[speciesIndex][i][j].adjFitness = self.speciesList[speciesIndex][i][j].fitness/totalNum
# [species1,species2,species3,etc.]
# each existing species is represented by a random genome inside the species from the previous generation
# 
	# The breed/repopulate function gets the genomes from the latest generation of the ith species
	# and assigns them to a temporary list; tempRepopList.
	# for each species:
	# for i in range (0,len(speciesList)):
		# assign the latest generation of that species to the temporary repopulation list
		# tempRepopList = self.speciesList[i][len(self.speciesList[i])-1]
		# create a new generation for the new population to go into
		# self.speciesList[i].append([])
	# (NOTE) that means I'll have to edit the champGenome function; so that it no longer adds a new generation to the ith species; as that's already being done.
		# if the ith species has more than *5* genomes, copy the best performing genome to the new generation
		# if self.genomeCount(i) > ** 5 **
	# If the best performing gene in the ith species is from its latest generation (not accounting for the new empty generation just added)
	# if champGenome.generationNum == speciesList[i][len(speciesList[i])-2[0]].generationNum: (cause we now have the new generation!)
	# remove it from the tempRepopList
		# tempRepopList.remove(inputGenomeOrWhateverItsCalled)

	# randomly choose 25% of the population to breed without crossover
	# tempNonCrossover = random.sample(tempRepopList, round(*0.25*))
	# ***optional*** copy the tempRepopList to tempCrossover list, and then remove the nonCrossover genomes from that tempCrossover list
	# ***or, you could go...***
	# remove any of the tempNonCrossover genomes from the tempRepopList
	# [tempRepopList.remove(i) for i  in tempNonCrossover]
	# for the number of remaining genomes in tempRepopList, randomly choose (with replacement), 2 genomes to crossBreed
		# append that child to the new generation of that species
	def breed(self, populationNum, mutWithoutCrossover, weightMutProb, peterbWeightProb, randWeightProb, complexifyProb, newConNum, newConProb, newNodeProb, childDisabProb, champGenomeThreshold, interSpeciesMatingRate, currentGeneration, lowWeight = -1, largeWeight = 1, decPlace = 2):
		
		# if the user wants to grow a genome through this new complexification method,
		if complexifyProb > 0.0:
			# disable the other expanding mechanism
			newNodeProb = 0.0
			newConProb = 0.0
		# NOTE: another patch to a serious problem; I need to solve the root issue.
		# This 'fixes' a problem where some genomes in the latest generation don't have a fitness score
		# This is a copy of the code chunk at the end of the function.
		# This loop removes any genomes from the second last generation of each species with no fitness value
		# (14/09/18)

		# index1 is for each species
		index1 = 0
		# index2 is for each generation in that species
		index2 = 0
		# index3 is for each genome in that generation
		index3 = 0

		# looping through each species
		while index1 < len(self.speciesList):
			index2 = 0
			# looping through each generation in that species
			# note the -1, as the latest generation won't have fitness values (as we intend). Also note that tempRepopList hasn't been made yet (15/09/18)
			while index2 < len(self.speciesList[index1])-1:
				index3 = 0
				# looping through each genome in that generation
				while index3 < len(self.speciesList[index1][index2]):
					# if there are any genomes with no fitness value, remove them
					if self.speciesList[index1][index2][index3].fitness == 'empty':
					# remove that non-simulated genome from that generation
						self.speciesList[index1][index2].remove(self.speciesList[index1][index2][index3])
					else:
						# move on to the next genome in that generation
						index3 += 1
				index2 += 1
			index1 += 1


		# NOTE: This deletes any generations with no genomes in them. (28/09/18)
		# I do this chunk at the end of the program too. Don't know why I'm having this error atm
		# looping through each species
		index1 = 0
		index2 = 0
		while index1 < len(self.speciesList):
			index2 = 0
			# looping through each generation in that species
			# note the -1, as the latest generation won't have fitness values (as we intend). Also note that tempRepopList hasn't been made yet (15/09/18)
			while index2 < len(self.speciesList[index1])-1:	
				if len(self.speciesList[index1][index2]) == 0:
					# remove that empty generation from that species
					self.speciesList[index1].remove(self.speciesList[index1][index2])
				else:
					# move on to the next generation in that species
					index2 += 1
			index1 += 1


		# reset the tempRepopList (idk if its useful)
		tempRepopList = []
		# for each species
		for i in range(0,len(self.speciesList)):
			print("we're looking at species number ", i)
			# assign the latest generation of that species to the temp repopulation list
			# Note the -2. This is the latest generation of the ith species with a fitness value. (2/09/18)
			tempRepopList = copy.copy(self.speciesList[i][len(self.speciesList[i])-2])
			# create a new generation for the new population to go into
			### Wait a minute. Why are we appending into the same species? We should simulate the new genomes before we speciate them (29/08/18)
			# self.speciesList[i].append([])
			# if our species has more than a set number of genomes, (default 5)
			# NOTE: genomeCount just tells us how many genomes there are. (30/08/18)
			# I am adding an argument to genomeCount that allows us to ignore a set number of generations from a species ( default 0 )
			if self.genomeCount(i,1) > champGenomeThreshold:
				# create a new generation for the champion genome to go into
				self.speciesList[i].append([])

				# copy the best performing genome
				largestFitnessTemp = copy.copy(self.largestFitness(i,1))
				# we're putting it in a new generation, so it should be given the appropriate number (6/09/18)
				largestFitnessTemp.generationNum = currentGeneration + 1
				# add it to the next generation of the species
				#self.speciesList[i][len(self.speciesList[i])-1].append(largestFitnessTemp)
				print('copying the best performing gene to the latest generation.')
				
				# Note the addition of the ,1 argument. This will agnore the latest generation (which doesn't have fitness values, which would raise errors later). (2/09/18)
				# NOTE: Adding in an extra thing (13/09/18)
				# If the best performing gene has already been added to the latest generation, don't add it again
				if largestFitnessTemp not in self.speciesList[i][len(self.speciesList[i])-1]:
					self.speciesList[i][len(self.speciesList[i])-1].append(self.largestFitness(i,1))


				# if the best performing gene is from the latest generation (not from the generation we just added!)
				# hence the -2 instead of -1
				# Note: trying -1 instead of -2
				if largestFitnessTemp in self.speciesList[i][len(self.speciesList[i])-1]:
				# 	# remove it from the temp repopulation list
					tempRepopList.remove(largestFitnessTemp)


			# What about these two lines below?? (7/09/18)
			# else:
			# else if the latest generation with genomes hasn't been simulated yet (no fitness score) (9/09/18)
			#elif self.speciesList[i][len(self.speciesList[i])-2]
			else:
				print('Adding a new generation to species ', i)
				self.speciesList[i].append([])

				# if the best performing gene is from the latest generation (not from the generation we just added!)
				# hence the -2 instead of -1
				#if largestFitnessTemp.generationNum == self.speciesList[i][len(self.speciesList[i])-2][0].generationNum:
				# if largestFitnessTemp in self.speciesList[i][len(self.speciesList[i])-2]:
				# # 	# remove it from the temp repopulation list
				# 	tempRepopList.remove(largestFitnessTemp)
				# temporarily disabling convenient line. some errors are coming up about x not in list. (1/09/18)
				#if self.largestFitness(i).generationNum == self.speciesList[i][len(self.speciesList[i])-3][0].generationNum:
				#	tempRepopList.remove(self.largestFitness(i))
			# randomly choose a set amount of the population to breed without crossover (default 25%)
			# note that our number of bred genomes is rounded up, as we can't select half a genome!
			tempNonCrossover = random.sample(tempRepopList, round(len(tempRepopList)*mutWithoutCrossover))
			# remove any non crossover genomes from our repopulation list
			#print('tempNonCrossover is: ', tempNonCrossover)
			#print('it is of type ',type(tempNonCrossover))
			#[print(i) for i in tempNonCrossover]
			[tempRepopList.remove(i) for i in tempNonCrossover]
			# for the remaining genomes in our repopulation list
			# THIS LINE BELOW IS RESPONSIBLE FOR THE POPULATION EXPLODING (30/08/18)
			# for the 75% of the population:

			newGenerationCount = 0
			prevGenerationCount = 0

			for j in range(0,len(tempRepopList)):
			# why dont you take the square root?
			# maybe I could just have the same number of children as parents
			#for j in range(0,round(len(tempRepopList)/2)):
			# To prevent population exploding through exponential growth, we could try the inverse? (1/09/18) 
		#for j in range(0,round((len(tempRepopList)**1/2)/2)):
		#for j in range(0,populationNum):
			#for j in range(0,1):

			# instead of breeding a child for each parent,
			# breed the same number of kids as the are parents?
				# maybe I could do it by the best two, then the next performing 2? that will make the system lose its variety though
				# randomly choose w/ replacement 2 genomes to crossBreed

				# maybe have a thing here where if the interspecies mating rate 0.001 is met,
				# the parentsTemp will be 1 parent from our current list and 1 parent from the current generation of another (randomly selected) species
				# interspecies functions like a boolean switch
				# for debugging
				# interSpeciesMatingRate = 1
				interspecies = random.choices([0,1], weights = [1-interSpeciesMatingRate,interSpeciesMatingRate])[0]
				# if it's met AND there's more than 1 species (you cant have interspecies mating with 1 species),
				if interspecies == 1 and len(self.speciesList) > 1:
					print("The parents will be from different species.")
					# one parent will be taken from the current generation of the ith species
					parentsTemp = random.sample(tempRepopList,1)
					#print('jeff 5: the parent genome from the repop list has fitness of ', parentsTemp[0].fitness)
					# another parent will be taken from the current generation of a different (randomly selected) species
					# a list of all the species indexes
					otherSpeciesList = [i for i in range(0,len(self.speciesList))]
					# we don't want to randomly choose the species we're already in (INTER-species mating), so remove it from the otherSpeciesList
					otherSpeciesList.remove(i)
					# randomly choose one of the other species
					interSpeciesIndex = random.choice(otherSpeciesList)
					# randomly choose one of the genomes in the latest generation of that species
					# note the -2. This is because we are taking from the latest species with a fitness value.(10/09/18)
					interSpeciesGenome = random.choice(self.speciesList[interSpeciesIndex][len(self.speciesList[interSpeciesIndex])-2])
					parentsTemp.append(interSpeciesGenome)
					#print('jeff 6: the other parent genome has fitness of ', parentsTemp[1].fitness)
				else:
					print('probably an error here.')
					print('Currently, these are all the species: ', self.speciesList)
					# Note the empty list in the latest generation. This is for all the new genomes to go into
					print('we would hope the first genomes in species ', i, ' have some fitness value before being bred; ', self.speciesList[i][0][0].fitness)
					#print('we would expect the latest genomes to have an empty fitness value: ', self.speciesList[i][len(self.speciesList[i])-2][0].fitness)
					print('This is the latest generation of species ', i, ':', self.speciesList[i][len(self.speciesList[i])-2])
					print('tempRepopList is: ', tempRepopList)
					print('fitness of tempRepopList[0] is', tempRepopList[0].fitness)
					# If we only have 0 or 1 genomes to work with
					if len(tempRepopList) < 2:
						# we have no more usable brains
					#	return "extinct"

					#Note: might be a problem here. What if a species doesn't get any genomes added to it??? (7/09/18)
						# A 'because I have to' addition (29/08/18)
						# make a copy of the current value and breed the child from that
						tempRepopList.append(copy.copy(tempRepopList[0]))
					#else:
					#	parentsTemp = random.sample(tempRepopList,2)

					# randomly chooses 2 parents from the latest generation of the ith species
					parentsTemp = random.sample(tempRepopList,2)
				# assign the child of 2 parents to a temp
				childTemp = mate.crossBreed(parentsTemp[0], parentsTemp[1], childDisabProb)
				#print('For error checking: the childTemp is ', childTemp)
				#print('the childTemp node genes are: ', childTemp[0])
				#print('the childTemp con genes are: ', childTemp[1])
				# NOTE: Adding the plus 1 after current Generation. The child will be of a newer generation! (6/09/18)
				childGenome = genome(childTemp[0], childTemp[1], generationNum = currentGeneration + 1)
				# add their child to the new generation of the ith species

			# The child belongs to one of the parent's species.
			# At the moment, it will be random, in case parents are from idfferent species.
			# It doesn't matter too much species; as long as it's placed in the latest generation of SOME species
				
				#NOTE: I'm testing a new idea out (6/09/18)
				# The child will belong together one of the parents; preferably make this to the stronger parent? idk
				# doesn't matter too much, cause after being simulated, they'll be appropriately placed into the proper species
				childSpecies = random.sample(parentsTemp,1)[0].speciesNum

				## JEFFFFFFF
			#self.speciesList[i][len(self.speciesList[i])-1].append(childGenome)
				#NOTE: Trying something new
				# If the latest genomes from a species have a smaller generation number, add a new generation and put the children/mutations in there
				# NOTE: what if I added the + 1 to generationNum? Our population goes [20,20,20,1]. Maybe some elements that should be grouped in the same generation aren't. 7/09/18
				
				# if our child genome is more advanced than its fitness-bearing cousins,
				# Note: trying out <= instead of < (8/09/18)
				if self.speciesList[childSpecies][len(self.speciesList[childSpecies])-2][0].generationNum < childGenome.generationNum:
				# 	#self.speciesList[childSpecies].append([])
					# add it into the same species in the latest gen
					self.speciesList[childSpecies][len(self.speciesList[childSpecies])-1].append(childGenome)
					newGenerationCount += 1
					print('TempRepopList: ', 'generationNum of the current genome is ', childGenome.generationNum, '. adding a new genome into the latest generation of species', childSpecies, '. We have done this adding process ', newGenerationCount, ' times.')

				# NOTE: trying out something. I'm noticing that after the first generation, only 75% of genomes go through. (13/09/18)
				else:
					# else, add the genome to the generation we're working with now. It's a useless line, I know.
					self.speciesList[childSpecies][len(self.speciesList[childSpecies])-2].append(childGenome)
					prevGenerationCount += 1
					print('TempRepopList: ', 'generationNum of the current genome is ', childGenome.generationNum, '. adding a new genome into the previous generation of species', childSpecies, '. We have done this adding process ', prevGenerationCount, ' times.')



				# elif self.speciesList[childSpecies][len(self.speciesList[childSpecies])-1][0].generationNum + 1 == childGenome.generationNum:
				# 	self.speciesList[childSpecies][len(self.speciesList[childSpecies])-1].append(childGenome)
			# for each of the genomes in the temp non-crossover list
		#for j in range(0,len(tempNonCrossover)-1):
		# for the 25%
				for j in range(0, len(tempNonCrossover)):
					print('Runs through tempNonCrossover: ', j+1)
				# there's a preset chance (default 3%) a new node will be added to a genome
					newNode = random.choices([0,1], weights = [1-newNodeProb, newNodeProb])
					# Note: I forgot that the output of random.choices is a list (13/09/18). thus, i've added [0] to the end of all the conditionals here
					if newNode[0] == 1:
						print('adding a new hidden node')
						tempNonCrossover[j].addNode()
				# there's a preset chance (default 30%) a new connection will be added to a genome
					newConnection = random.choices([0,1], weights = [1-newConProb, newConProb])
					if newConnection[0] == 1:
					# randomly add a connection gene to the jth genome in the list
					# NOTE
					# maybe add an argument for the range the new connection weight could be?
						print('adding a random connection')
						tempNonCrossover[j].addCon()

					# NOTE: remembering that if the user enables the complexifyProb, the newNode and newCon probabilities are disabled
					# That means that the if statements in the above code chunks will not be met (1/10/18)
					complexify = random.choices([0,1], weights = [1-complexifyProb, complexifyProb])
					if complexify[0] == 1:
						# add 1 new node.
						tempNonCrossover[j].addNode()
						# remembering the -1 due to newNodes being 'superimposed' onto previous connections.
						# NOTE: I'm trying out different values on the denominator; The more connections added, the longer it takes to do a simulation
						# round(158/4) = 40 
						#for k in range(0, int(round(len(tempNonCrossover[j].sensor)-1)/4)):
						for k in range(0, newConNum):
							tempNonCrossover[j].addCon()

				# There is a preset chance (default 80%) a genome has its connection weights mutated
					weightMut = random.choices([0,1], weights = [1-weightMutProb, weightMutProb])
				# If it does:
					if weightMut[0] == 1:
					# There is a preset chance (default 90%) that a genome's connection weights are uniformly peterbed
						peterbWeight = random.choices([0,1], weights = [1-peterbWeightProb, peterbWeightProb])
						if peterbWeight[0] == 1:
						# this is like the randWeight function in the genome class.
						# the difference is that the variance of random values could be unique to each genome,
						# while here, every genome in every species gets uniformly peterbed with the same variance
						# for each of the connection genes
							for k in range(0,len(tempNonCrossover[j].conGenes)):
								tempNonCrossover[j].conGenes[k][2] += round(random.uniform(lowWeight, largeWeight), decPlace)
							print('each weight is being uniformly peterbed')
						#[k[2] += round(random.uniform(lowWeight, largeWeight), decPlace) for k in tempNonCrossover[j].conGenes]
						# uniformly peterb all of that genome's conenction weights by a specified range
						else:
						# assign each of the genome's connection weights a new random value (in a specified range)
						# each genome could have its own specified variance.

							for k in range(0,len(tempNonCrossover[j].conGenes)):
								tempNonCrossover[j].conGenes[k][2] = tempNonCrossover[j].randWeight(decPlace = decPlace) 
							print('each weight is being assigned a new random value')
						# the weights (k[2]) of each of the connection genes becomes a random weight
						#[k[2] = tempNonCrossover[j].randWeight() for k in tempNonCrossover[j].conGenes]

					# NOTE: trying to patch a problem where no enabled connection genes appear in future generations (around generation 21) (24/09/18)\
					# Maybe a for loop? for each connection gene, if it's innovation nomber doesn't appear in the current list of enabConGenes and enabled == 1?
					# maybe just a simple: add each of the connection genes of which enabled == 1?
					for k in range(0, len(tempNonCrossover[j].conGenes)):
						# Note: simulation was slowing down HEAPS! So I added this line to check that genes aren't added more than once. (25/09/18)
						# if a connection gene hasn't already been added to the enabledConnections list
						# (if there are no genes in enabConGenes) or (if a gene's innovation number doesn't appear in in enabConGenes)
						# type(tempNonCrossover[j].enabConGenes) == int or 
						if tempNonCrossover[j].conGenes[k][4] not in [i[4] for i in tempNonCrossover[j].enabConGenes]:
						# if a connection gene is enabled,
							if tempNonCrossover[j].conGenes[k][3] == 1:
							# Add it to the list of enabled connection genes
								tempNonCrossover[j].enabConGenes.append(tempNonCrossover[j].conGenes[k])

						# if it doesn't work, I can add a check to see if the innovation number of that gene already appears in the enabConGenes list


			# put the tempNonCrossover genomes into the latest generation of the ith species
			#[print(j) for j in tempNonCrossover]
		#[self.speciesList[i][len(self.speciesList[i])-1].append(j) for j in tempNonCrossover]
			print('Now that every genome in tempNonCrossover has been transformed, onto placing.')
		# Note: added this (6/09/18)
		# set each genome's generationNum as currentGeneration + 1
			newGenerationCount = 0
			prevGenerationCount = 0
			for j in tempNonCrossover:
				j.generationNum = currentGeneration + 1
				#NOTE, trying something new here (13/09/18)
				#j.generationNum += 1
				# if our given genome comes from a species with a smaller current generation number,
				# Note: trying out <= instead of < (8/09/18)
				if self.speciesList[j.speciesNum][len(self.speciesList[j.speciesNum])-2][0].generationNum < j.generationNum:
					# add a new generation to that species
					#self.speciesList[j.speciesNum].append([])
					# put our given genome in the latest generation of that species
					self.speciesList[j.speciesNum][len(self.speciesList[j.speciesNum])-1].append(j)
					newGenerationCount += 1
					print('TempNonCrossover: adding a new genome into the latest generation of species', j.speciesNum, '. We have done this ', newGenerationCount, ' times.')
				# another useless line, I know.
				else:
					self.speciesList[j.speciesNum][len(self.speciesList[j.speciesNum])-2].append(j)
					prevGenerationCount += 1
					print('TempNonCrossover: adding a new genome into the previous generation of species', j.speciesNum, '. We have done this ', prevGenerationCount, ' times.')
				# else if our given genome from a species' latest generation is the same as our given genome (aka, if genomes have already been added in for this generation,)
				# elif self.speciesList[j.speciesNum][len(self.speciesList[j.speciesNum])-1][0].generationNum + 1 == j.generationNum:
				# 	# put our given genome in the latest generation of that species
				# 	self.speciesList[j.speciesNum][len(self.speciesList[j.speciesNum])-1].append(j)
			print('Now that every genome in tempNonCrossover has been placed, onto removing empty genomes')
			#[self.speciesList[j][len(self.speciesList[j.speciesNum])-1].append(j) for j in tempNonCrossover]

				# I might do a shuffle/permutation of the output, for good measure. not sure yet though.
			random.shuffle(self.speciesList[i][len(self.speciesList[i])-1])

			#NOTE: There's a problem where occasionally, the latest generation will be an empty list, and nothing will be put in it
			# This just checks if there are any empty lists at the end, and then removes those empty lists
			# I know, it's only a temporary solution to a fundamental problem with this function, but it's better than nothing (9/09/18)

			#NOTE 2: I think I figured out the problem. 
			# after a round of simulations, there might be some species that dont get any of the new genomes
			# my code automatically adds a new generation each time; but if there are times where there are no new genomes, there'll be empty lists! that's a bad thing.

		# index1 is for each species
		index1 = 0
		# index2 is for each generation in that species
		index2 = 0

			# looping through each species
		while index1 < len(self.speciesList):
			index2 = 0
			# looping through each generation in that species
			while index2 < len(self.speciesList[index1]):
				# if there are any generations with no elements in them (an empty list), remove them
				if len(self.speciesList[index1][index2]) == 0:
					# remove that blank generation from that species
					self.speciesList[index1].remove(self.speciesList[index1][index2])
				else:
					# move on to the next generation
					index2 += 1
			index1 += 1

		#NOTE: trying something new here. There was a problem where all the new simulaated genomes are placed in the same genome as the parents. (13/09/18)
		# My solution is to make a new list for our new population to go in
		# for each species
		# for i in range(0,len(self.speciesList)):
		# 	# if the latest generation is already a blank list (aka, no genomes have been added in during that simulation run),
		# 	if len(self.speciesList[i][len(self.speciesList[i])-1]) == 0:
		# 		print('The latest generation is already empty. Skip')
		# 		# skip it
		# 		pass
		# 	# else, add a new generation (empty list) onto the ith species
		# 	else:
		# 		print('Adding a new generation to species ', i)
		# 		self.speciesList[i].append([])


		# NOTE: another patch to a serious problem; I need to solve the root issue
		# This loop removes any genomes from the second last generation of each species with no fitness value
		# (9/09/18)

		# index1 is for each species
		index1 = 0
		# index2 is for each generation in that species
		index2 = 0
		# index3 is for each genome in that generation
		index3 = 0

		# looping through each species
		while index1 < len(self.speciesList):
			index2 = 0
			# looping through each generation in that species
			# note the -1, as the latest generation won't have fitness values (as we intend)
			while index2 < len(self.speciesList[index1])-1:
				index3 = 0
				# looping through each genome in that generation
				while index3 < len(self.speciesList[index1][index2]):
					# if there are any genomes with no fitness value, remove them
					if self.speciesList[index1][index2][index3].fitness == 'empty':
					# remove that non-simulated genome from that generation
						self.speciesList[index1][index2].remove(self.speciesList[index1][index2][index3])
					else:
						# move on to the next genome in that generation
						index3 += 1
				index2 += 1
			index1 += 1

# NOTE: I'm getting a problem at the end of generation 4.
# It's consistently been about the latest generation of species 2 (the latest species, in my examples)
# the latest generation is just an empty list.
# is that because genomes that are *supposed* to go in there aren't?
# is it an unneccesary list?
# does this have to do with the 'champion genome' function?

# Error classes


class error(Exception):
	pass

class neatError(error):
	""" 
	Message -- explanation of the error
	"""
	def __init__(self, message):
		self.message = message
