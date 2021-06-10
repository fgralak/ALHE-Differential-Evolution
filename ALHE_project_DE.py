#!/usr/bin/env python3

import numpy as np
import math

# load basic population, seed list and parameters
pop = np.loadtxt('M_1_D10.txt')
seeds = np.loadtxt('Rand_Seeds.txt')
popsize = len(pop)
dimensions = len(pop[0])
func_no = 1
Runs = 30
run_id = 1
f = 0.8
cr = 0.7
maxFES = 200000
errors = []

# differential evolution algorithm DE/rand/1/bin

# reference algorithm
def de_ref():
	curFES = 0
	catch_error = 0
	while curFES < maxFES:
		for j in range(popsize):
			p_selected = pop[select_rand(popsize)]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			cross = crossover(pop[j], mutate, cr)
			pop[j], curFES = tournament(pop[j], cross, curFES)
	print(pop)

# varaint A
def de_A(): 
	curFES = 0
	while curFES < maxFES:
		for j in range(popsize):
			winner, curFES = select_tour(pop, popsize, curFES)
			p_selected = pop[winner]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			cross = crossover(pop[j], mutate, cr)
			pop[j], curFES = tournament(pop[j], cross, curFES)
	print(pop)

# varaint B
def de_B():
	curFES = 0
	while curFES < maxFES:
		for j in range(popsize):
			p_selected = pop[select_rand(popsize)]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			pop[j] = crossover(pop[j], mutate, cr)
	print(pop)

# varaint C
def de_C(): 
	curFES = 0
	while curFES < maxFES:
		for j in range(popsize):
			winner, curFES = select_tour(pop, popsize, curFES)
			p_selected = pop[winner]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			pop[j] = crossover(pop[j], mutate, cr)
	print(pop)

# auxiliary functions

# select a point randomly
def select_rand(popsize):
	return np.random.randint(popsize)

# select a point using tournament
def select_tour(pop, popsize, curFES):
	p1 = np.random.randint(popsize)
	p2 = np.random.randint(popsize)
	fitness1 = pick_fun(func_no, pop[p1], curFES)
	fitness2 = pick_fun(func_no, pop[p2], curFES)
	if fitness1 < fitness2: 
		return p1, curFES
	else: 
		return p2, curFES

# keeping points inside bounds
def bounds(mutate):
	bounded = []
	for m in mutate:
		if m > 100:
			m = 100
		elif m < -100:
			m = -100
		bounded.append(m)
	return bounded

# binary crossover
def crossover(original, mutate, cr):
	cross = []
	for o, m in zip(original, mutate):
		if np.random.rand() < cr:
			cross.append(m)
		else:
			cross.append(o)
	return cross

# tournament that tells which point should be in new population 
def tournament(original, cross, curFES):
	fitness1, curFES = pick_fun(func_no, original, curFES)
	fitness2, curFES = pick_fun(func_no, cross, curFES)
	if fitness1 < fitness2: 
		return original, curFES
	else: 
		return cross, curFES

# setting seed for random function
def set_seed(dimensions, func_no, Runs, run_id):
	seed_ind = (dimensions / 10 * func_no * Runs + run_id) - Runs
	seed_ind = seed_ind % 1000
	np.random.seed(int(seeds[int(seed_ind)]))

# picking corrent fitness function
def pick_fun(func_no, individual, curFES):
	if func_no == 1:
		fitness_value = bent_cigar_function(individual)
	elif func_no == 2:
		fitness_value = rastrigins_function(individual)
	elif func_no == 3:
		fitness_value = high_conditioned_elliptic_function(individual)
	elif func_no == 4:
		fitness_value = hgbat_function(individual)
	elif func_no == 5:
		fitness_value = rosenbrocks_function(individual)
	elif func_no == 6:
		fitness_value = griewanks_function(individual)
	elif func_no == 7:
		fitness_value = ackleys_function(individual)
	elif func_no == 8:
		fitness_value = happycat_function(individual)
	elif func_no == 9:
		fitness_value = discus_function(individual)
	else:
		print('ERROR')
	curFES += 1
	for i in range(16):
		if error_points[i] == curFES:
			errors.append(fitness_value)
	return fitness_value, curFES

# calculate points where we need to save error value
def calculate_error_points(dimensions):
	error_points = []
	for k in range(16):
		error_points.append(int(pow(dimensions, (k/5)-3) * maxFES))
	return error_points

# function definitions

# definition of basic function 1
def bent_cigar_function(individual):
	fitness_value = individual[0]**2
	for i in range(1,dimensions):
		fitness_value += individual[i]**2 * 10**6
	return fitness_value

# definition of basic function 2
def rastrigins_function(individual):
	fitness_value = 0
	for i in range(dimensions):
		fitness_value += individual[i]**2 + 10 - (10 * math.cos(2 * math.pi * individual[i]))
	return fitness_value

# definition of basic function 3
def high_conditioned_elliptic_function(individual):
	fitness_value = 0
	for i in range(dimensions):
		temp = (i)/(dimensions-1)
		fitness_value += 10**(6*temp) * individual[i]**2 
	return fitness_value

# definition of basic function 4
def hgbat_function(individual):
	fitness_value = 0
	square = 0
	summ = 0
	for i in range(dimensions):
		square += individual[i]**2
		summ += individual[i]
	fitness_value = abs(square**2 - summ**2)**(1/2) + (0.5 * square + summ)/dimensions + 0.5
	return fitness_value	

# definition of basic function 5
def rosenbrocks_function(individual):
	fitness_value = 0
	for i in range(dimensions-1):
		fitness_value += 100 * (individual[i]**2 - individual[i+1])**2 + (individual[i] - 1)**2 
	return fitness_value

# definition of basic function 6
def griewanks_function(individual):
	fitness_value = 0
	temp = 1
	for i in range(dimensions):
		fitness_value += individual[i]**2 / 4000
		temp *= math.cos(individual[i]/((i+1)**(1/2)))
	fitness_value = fitness_value - temp + 1
	return fitness_value

# definition of basic function 7
def ackleys_function(individual):
	fitness_value = 0
	temp1 = 0
	temp2 = 0
	for i in range(dimensions):
		temp1 += individual[i]**2
		temp2 += math.cos(2 * math.pi * individual[i])
	temp1 = -0.2 * ((temp1 / dimensions)**(1/2))
	temp2 = temp2 / dimensions	
	fitness_value = -20 * math.exp(temp1) - math.exp(temp2) + 20 + math.e 
	return fitness_value

# definition of basic function 8
def happycat_function(individual):
	fitness_value = 0
	square = 0
	summ = 0
	for i in range(dimensions):
		square += individual[i]**2
		summ += individual[i]	
	fitness_value = abs(square - dimensions)**(1/4) + (0.5 * square + summ)/dimensions + 0.5
	return fitness_value

# definition of basic function 9
def discus_function(individual):
	fitness_value = 10**6 * individual[0]**2
	for i in range(1,dimensions):
		fitness_value += individual[i]**2
	return fitness_value

# calling functions


set_seed(len(pop[0]), func_no, Runs, run_id)	
error_points = calculate_error_points(dimensions)

de_ref()
#de_A(lambda x: np.exp(x))
#de_B(lambda x: np.exp(x))
#de_C(lambda x: np.exp(x))