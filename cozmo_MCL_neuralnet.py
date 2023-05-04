import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

import sys
import os
import cv2
import random
import math
import PIL
import pandas as pd     
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import util
import util_robot as util_r

import sys
sys.path.append('../')
from model.utils import util as util_m
from model.model.model import MyModel
from model.dataset.dataset import MyDataset

# Arbitrary values, to model gaussian noise.
sensorVariance = 0.01
proportionalMotionVariance = 0.01
degree_increment = 5.0
DATA_DIR = './model/data/0'

def get_infer_img(img_path: str):
  img = np.asarray(PIL.Image.open(img_path))
  img = np.stack([img, img, img], axis=-1)
  img = torch.from_numpy(img).permute(2, 0, 1).to(cfg['device']) / 255.0
  return img.unsqueeze(0)  # (c, h, w)

def get_emb(model, img):
  with torch.no_grad():
    emb = model.forward_one(img)
    return emb.squeeze(0).detach().cpu().numpy()

def euclidean_dist(a, b) -> np.ndarray:
  ''' a, b have shape=(length, )
  '''
  return np.linalg.norm(a-b, axis=0)

def compare_embs(emb1, emb2):
  return euclidean_dist(emb1, emb2)

def compare_images(emb1, emb2):
  img1 = emb1/255.0
  img2 = emb2/255.0
  return ((img1 - img2)*(img1 - img2)).sum()

def measurement_model(particlePose, currID, img_list):
  # Get the source image (from initial rotation) to compare to by rounding to the nearest multiple of degree_increment
  roundedPose = int(degree_increment * round(float(particlePose) / degree_increment)) % 360 # assume degree_increment = 5
  img_id = int(roundedPose // degree_increment)
  diff = compare_images(img_list[f'{img_id}.jpg'], img_list[f'{currID}.jpg'])
  #see Text Table 5.2, implementation of probability normal distribution
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (diff * diff) / (2 * sensorVariance))

def measurement_model_nnet(particlePose, currID, emb_list):
  roundedPose = int(degree_increment * round(float(particlePose) / degree_increment)) % 360 # assume degree_increment = 5
  img_id = int(roundedPose // degree_increment)
  emb_diff = compare_embs(emb_list[f'{img_id}.jpg'], emb_list[f'{currID}.jpg'])
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (emb_diff * emb_diff) / (2 * sensorVariance))

def motion_model(movement, current_position):
  # making variance proportional to magnitude of motion command
  newDeg = current_position - movement - sample_normal_distribution(abs(movement * proportionalMotionVariance)) 
  return newDeg % 360

def sample_normal_distribution(variance):
  total = 0
  for _ in range(12):
    total += (2.0 * random.random()) - 1.0
  return math.sqrt(variance) * total / 2.0

def MCL():
  cfg = util_m.load_cfg('./model/config/configuration.json')
  cfg['data_dir'] = './model/data'
  # model = get_model(cfg)

  img_name_list = [img_file for img_file in os.listdir(DATA_DIR) if img_file.endswith('jpg') and img_file[0].isdigit()]

  img_list = {img_name: cv2.imread(os.path.join(DATA_DIR, img_name), 0) for img_name in img_name_list}
  # img_infer_list = {img_name: get_infer_img(os.path.join(DATA_DIR, img_name)) for img_name in img_name_list}
  # emb_list = {img_name: get_emb(model, img_infer_list[img_name]) for img_name in img_infer_list.keys()}


  M = 200
  particles = np.random.randint(0, 360, (M,)) # [random.randint(0, 360) for _ in range(M)] 

  curr_id = 21

  for _ in range(20):
    # take new image
    curr_id -= 1

    # Initialize arrays to store poses, corresponding weights, and their normalized probabilities
    poses = np.array([])
    weights = np.array([])

    # for each potential position
    for p in range(M):
      currentPosition = particles[p]
      # update our belief about where the given pose represents, given the movement just made
      new_pose = motion_model(degree_increment, currentPosition)
      # Assign a weight to this position based on the image difference
      weight = measurement_model(new_pose, curr_id, img_list) 
      # store this information
      poses = np.append(poses, new_pose)
      weights = np.append(weights, weight)

    probs = weights/np.sum(weights)
    cdf = np.sum(np.tril(probs), axis=1) 
    assert abs(cdf[-1] - 1.0) < 1e-8, 'last index in CDF must be 1.0'
    
    # Resample, according to this CDF
    new_particles = []
    for p in range(M):
      p = random.random()
      my_index = 0
      while p >= cdf[my_index]:
        my_index += 1
      new_particles.append(poses[my_index])

    # Specify the new population of positions for the next iteration
    particles = np.array(new_particles)

    # visualize the robot's beliefs about it's current position
    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.hist(np.array(newParticles))
    # plt.show()

    # Sum up the belief probabilities, in 20 degree increments    
    bin_width = 20
    prob_bins = [0 for i in range(0, 360, bin_width)]
    for i in range(M):
      id = int(poses[i] // bin_width)
      prob_bins[id] += probs[i]

    # Print an estimated positoin
    max_prob_bin = max(prob_bins)
    if max_prob_bin != 0:
      est_pos = np.argmax(prob_bins) * bin_width
      print(f'my_est_position: {est_pos}')

    print(f'The 20 degree bin with the higher probability has a probability of {max(prob_bins)}')
    # update the probability in the max bin so the robot can continue if it is still unsure
    max_prob_bin = max(prob_bins)

    # based on the position the robot thinks it is in, rotate back to home
    # robot.turn_in_place(degrees(-est_position)).wait_for_completed()
    # robot.say_text("I'm hooooooooome!").wait_for_completed()
    print('my_est_position', est_pos, '\tcurr_pos', (curr_id*degree_increment))

def get_model(cfg):
  model = MyModel(modeltype='resnet18', emb_size=cfg['emb_size'])
  model_weight_path = f'./model/saved/best_{model.modeltype}.pt'
  if os.path.exists(model_weight_path):
      model.load_state_dict(torch.load(model_weight_path))
      print('loaded saved model')
  model.to(cfg['device'])
  print('model created, type=', model.modeltype)
  return model

if __name__ == '__main__':
  cfg = util_m.load_cfg('./model/config/configuration.json')
  cfg['data_dir'] = './model/data'


  print()
  MCL()
