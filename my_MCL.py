'''
Credit belongs to https://www.youtube.com/watch?v=JhkxtSn9eo8&ab_channel=MohammadAltaleb
'''

import numpy as np

def sense_update(currP: np.ndarray, Z, all_particles: list) -> np.ndarray:
	'''
	Args:
		currP: probability of location (1D array)
		Z: data from sensor
		all_particles: list of all data points from environment
	Return:
		newP: probability of location updated (1D array)
	'''
	newP = []
	for i in range(len(currP)):
		hit = int(Z == all_particles[i])
		newP.append(currP[i] * (hit * pHit + (1 - hit) * pMiss))
	newP = np.array(newP)
	return newP/np.sum(newP)	# normalize

def move_update(currP: np.ndarray, move_step: int) -> np.ndarray:
	'''
	Args:
		currP: probability of location before moving (1D array)
		move_step: amount of movement; if step > 0 move to right, if step > 0 move to left
	Return:
		newP: probability of location after moving for "step" times (1D array)
	'''
	size = len(currP)
	newP = [currP[(i - move_step) % size] for i in range(size)]
	return np.array(newP)

def MCLocalize(all_particles: list, move_step: int, measurement) -> np.ndarray:
	'''
	Args:
		all_particles: list of all data points from environment
		move_step: amount of movement; if step > 0 move to right, if step > 0 move to left
		measurements: data from sensor
	Return:
		prob: probability of location (1D array)
	'''
	n = len(all_particles)
	prob = np.ones(n) / n

	prob = move_update(prob, move_step)
	prob = sense_update(prob, measurement, all_particles)
	
	return prob
	
if __name__ == '__main__':
	# the world to operate in
	all_particles = ['green', 'red', 'red', 'green', 'green']
	
	# multiply the probability by pHit if it correspond to the measurement
	pHit = 0.6
	# multiply the probability by pMiss if it does not correspond to the measurement
	pMiss = 0.2
	
	moveRight = 1
	moveLeft = -1
	
	# testing the functions
	measurements = ['red', 'red', 'green']
	
	p = MCLocalize(all_particles, moveRight, measurements)
	print('final result', [round(val, 2) for val in p])

