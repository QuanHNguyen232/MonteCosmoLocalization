'''
Credit belongs to https://www.youtube.com/watch?v=JhkxtSn9eo8&ab_channel=MohammadAltaleb
'''

import os
import random
import numpy as np

import img_processing as imgPr

def move_update(currP: np.ndarray, move_step: int) -> np.ndarray:
	size = len(currP)
	newP = [currP[(i - move_step) % size] for i in range(size)]
	return np.array(newP)

def sense_update(currP: np.ndarray, currData: np.ndarray, all_particles: list,) -> np.ndarray:
	newP = np.zeros_like(currP)
	size = currData.reshape(-1)[0]
	for i in range(len(all_particles)):
		diff = np.power(all_particles[i] - currData, 2).sum()
		newP[i] = diff / size
	newP = 1.0 - newP/np.sum(newP)	# for MSE, the lower the better --> 1 - lower = larger likelihood
	newP = np.multiply(currP, newP/np.sum(newP))	# update based on old prob
	return newP/np.sum(newP)	# normalize again

def MCLocalize(prob: np.ndarray, all_particles: list, move_step: int, measurement) -> np.ndarray:
	prob = move_update(prob, move_step)
	prob = sense_update(prob, measurement, all_particles)
	return prob

def sense_update_old(currP: np.ndarray, Z, all_particles: list) -> np.ndarray:
	newP = []
	pHit, pMiss = 0.6, 0.2
	for particle in all_particles:
		hit = int(Z == particle)
		newP.append(hit * pHit + (1 - hit) * pMiss)
	newP = np.array(newP)
	newP = newP/np.sum(newP)	# convert to prob (normalize)
	newP = np.multiply(currP, newP)	# update based on old prob
	return newP/np.sum(newP)	# normalize

def MCLocalize_old(prob, all_particles: list, move_step: int, measurement) -> np.ndarray:
	prob = move_update(prob, move_step)
	prob = sense_update_old(prob, measurement, all_particles)
	return prob


if __name__ == '__main__':
	# the world to operate in
	# all_particles = ['green', 'red', 'red', 'green', 'green']
	# measurements = ['green', 'red', 'red']

	# n = len(all_particles)
	# prob = np.ones(n) / n
	# for i in range(len(measurements)):
	# 	prob = MCLocalize_old(prob, all_particles, 1, measurements[i])
	# print('final result', [round(val, 2) for val in prob])

##########################################################

	IMG_DIR = 'cozmo-images-kidnap'

	all_particles = []
	for i in range(20):
		imgname = f'{IMG_DIR}/{i}-{i*18.0}.jpg'
		all_particles.append(imgPr.normalize_img(imgPr.get_img(imgname)))
	
	print('num_imgs', len(all_particles))
	print()

	n = len(all_particles)
	prob = np.ones(n) / n

	num_rotate = 3
	random_loc = random.randint(1, len(all_particles) - num_rotate)

	for i in range(num_rotate):
		random_imgname = f'{IMG_DIR}/{random_loc+i}-{(random_loc+i)*18.0}.jpg'
		currData = imgPr.normalize_img(imgPr.get_img(random_imgname))
		prob = MCLocalize(prob, all_particles, 1, currData)

		# print([round(val, 3) for val in prob])
		# print('actual loc', random_loc+i)
		# print('predict loc', prob.argmax())
		# print()
	
