'''
Credit belongs to https://www.youtube.com/watch?v=JhkxtSn9eo8&ab_channel=MohammadAltaleb
'''

import os
import random
import numpy as np

import img_processing as imgPr
from img_processing import get_kernel, convolution

def mse(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
	size = data2.reshape(-1)[0]
	res = np.power(data1 - data2, 2).sum()
	return res/size

def cos_similar(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
	data1, data2 = data1.reshape(-1), data2.reshape(-1)
	return np.dot(data1, data2) / (np.linalg.norm(data1) * np.linalg.norm(data2))

def normalize_prob(prob: np.ndarray) -> np.ndarray:
	return prob/np.sum(prob)

def move_update(currP: np.ndarray, move_step: int) -> np.ndarray:
	size = len(currP)
	newP = [currP[(i - move_step) % size] for i in range(size)]
	return np.array(newP)

def sense_update(currP: np.ndarray, data: np.ndarray, particles: list, method: str='mse') -> np.ndarray:
	newP = np.zeros_like(currP)
	for i, ptc in enumerate(particles):
		newP[i] = mse(ptc, data) if method == 'mse' else cos_similar(ptc, data)
	if method == 'mse': newP = 1.0 - normalize_prob(newP)	# for MSE, the lower the better --> 1 - lower = larger likelihood
	newP = normalize_prob(np.multiply(currP, newP))	# update based on old prob
	return newP

def MCLocalize(prob: np.ndarray, particles: list, move_step: int, measurement: np.ndarray, method: str='mse') -> np.ndarray:
	prob = move_update(prob, move_step)
	prob = sense_update(prob, measurement, particles, method)
	return prob

#Old MCL ##################################
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
	kernel_type = 'x'
	img_process = lambda x : convolution(x, get_kernel(kernel_type), stride=1, pad=1)

	all_particles = []
	num_imgs = 20
	for i in range(num_imgs):
		imgname = f'{IMG_DIR}/{i}-{i*(360.0/num_imgs)}.jpg'
		img = imgPr.normalize_img(imgPr.get_img_gray(imgname))
		all_particles.append(img)
	all_particles = [img_process(img) for img in all_particles]

	print('num_imgs', len(all_particles))
	print()

	n = len(all_particles)
	prob = np.ones(n) / n

	num_rotate = 1
	random_loc = random.randint(1, len(all_particles) - num_rotate)
	# random_loc = 0

	for i in range(num_rotate):
		# random_imgname = f'{IMG_DIR}/{random_loc+i}-{(random_loc+i)*(360.0/num_imgs)}.jpg'
		random_imgname = f'{IMG_DIR}/kidnapPhoto.jpg'
		currData = imgPr.normalize_img(imgPr.get_img_gray(random_imgname))
		prob = MCLocalize(prob, all_particles, 1, img_process(currData), 'mse')

		# print([round(val, 5) for val in prob])
		if random_imgname==f'{IMG_DIR}/kidnapPhoto.jpg': print('actual loc 8 or 9', end='\t')
		else: print('actual loc', random_loc+i, end='\t')
		print('predict loc', prob.argmax())

