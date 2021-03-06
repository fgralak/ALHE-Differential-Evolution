#!/usr/bin/env python3
import statistics

import numpy as np
import math

DIMENSIONS = 20
FUNCTION_NUMBER = 1
ROTATION = False
SHIFT = False
BIAS = False

# load basic population, seed list and parameters
pop = np.loadtxt(f'input_data/M_{FUNCTION_NUMBER}_D{DIMENSIONS}.txt')
pop_next = pop.copy()
seeds = np.loadtxt('input_data/Rand_Seeds.txt')
popsize = len(pop)
dimensions = len(pop[0])
func_no = FUNCTION_NUMBER
Runs = 30
run_id = 1
f = 0.7
cr = 0.9
maxFES = 200000
errors = []

M = np.loadtxt(f'input_data/M_{FUNCTION_NUMBER}_D{DIMENSIONS}_nr.txt') if ROTATION else np.identity(dimensions)
oi = np.loadtxt(f'input_data/shift_data_{FUNCTION_NUMBER}.txt')[:dimensions] if SHIFT else np.zeros(dimensions)

Fi = [100, 1100] if BIAS else [0, 0]

# differential evolution algorithm DE/rand/1/bin

# reference algorithm
def de_ref(pop, pop_next):
	curFES = 0
	catch_error = 0
	while curFES < maxFES:
		if len(errors) != 0:
			if errors[-1] < 10**(-8):
				break

		for j in range(popsize):
			p_selected = pop[select_rand(popsize)]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			cross = crossover(pop[j], mutate, cr)
			pop_next[j], curFES = tournament(pop[j], cross, curFES)
		pop = pop_next.copy()
	# print(pop_next)

# varaint A
def de_A(pop, pop_next):
	curFES = 0
	while curFES < maxFES:
		if len(errors) != 0:
			if errors[-1] < 10**(-8):
				break

		for j in range(popsize):
			winner, curFES = select_tour(pop, popsize, curFES)
			p_selected = pop[winner]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			cross = crossover(pop[j], mutate, cr)
			pop_next[j], curFES = tournament(pop[j], cross, curFES)
		pop = pop_next.copy()
	# print(pop_next)

# varaint B
def de_B(pop, pop_next):
	curFES = 0
	while curFES < maxFES:
		for j in range(popsize):
			p_selected = pop[select_rand(popsize)]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			pop_next[j] = crossover(pop[j], mutate, cr)
			temp = bent_cigar_mod(pop[j])
			print(temp)
		pop = pop_next.copy()
	print(pop_next)

# varaint C
def de_C(pop, pop_next):
	curFES = 0
	while curFES < maxFES:
		if len(errors) != 0:
			if errors[-1] < 10**(-8):
				break

		for j in range(popsize):
			winner, curFES = select_tour(pop, popsize, curFES)
			p_selected = pop[winner]
			p2 = pop[select_rand(popsize)]
			p3 = pop[select_rand(popsize)]
			mutate = p_selected + f * (p2 - p3)
			mutate = bounds(mutate)
			pop_next[j] = crossover(pop[j], mutate, cr)
			temp = bent_cigar_mod(pop[j])
			print(temp)
		pop = pop_next.copy()
	print(pop_next)

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
	func = functions[func_no]
	fitness_value = func(individual)
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
	for i in range(1, dimensions):
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
		temp = i / (dimensions-1)
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

# definition of basic function 11

def modified_schwefels_function(individual):
	fitness_value = 0
	for i in range(dimensions):
		zi = individual[i] + 4.209687462275036 * math.e + 2
		if abs(zi) <= 500:
			gzi = zi * math.sin(abs(zi)**(1/2))
		elif zi > 500:
			gzi = (500 - zi % 500) * math.sin((abs(500 - zi % 500)**(1/2)) - ((zi - 500)**(1/2)) / (10000 * dimensions))
		else:
			gzi = (abs(zi) % 500 - 500) * math.sin((abs(abs(zi) % 500 - 500))**(1/2)) - ((zi + 500)**2) / (10000 * dimensions)

		fitness_value -= gzi
	return fitness_value + 418.9829 * dimensions

# Modified functions

# 1
def bent_cigar_mod(individual):
	return bent_cigar_function(M.dot(individual) - oi) + Fi[0]

# 2
def schwefels_mod(individual):
	return modified_schwefels_function(M.dot(10 * (individual - oi))) + Fi[1]


functions = {
	1: bent_cigar_mod,
	2: schwefels_mod,
	3: high_conditioned_elliptic_function,
	4: hgbat_function,
	5: rosenbrocks_function,
	6: griewanks_function,
	7: ackleys_function,
	8: happycat_function,
	9: discus_function,
}

# calling functions

set_seed(len(pop[0]), func_no, Runs, run_id)
error_points = calculate_error_points(dimensions)

all_errors = []

for i in range(1, 31):
	run_id = i
	errors = []
	pop = np.loadtxt( f'input_data/M_{FUNCTION_NUMBER}_D{DIMENSIONS}.txt' )
	pop_next = pop.copy()
	# de_ref(pop, pop_next)
	de_A( pop, pop_next )
	# de_B(pop, pop_next)
	# de_C(pop, pop_next)
	print( errors[ -1 ] )
	all_errors.append(errors[-1])

print(all_errors)
print(f"max = {max(all_errors)}\nmin = {min(all_errors)}\navg = {sum(all_errors) / len(all_errors)}\nmedian = {statistics.median(all_errors)}")

