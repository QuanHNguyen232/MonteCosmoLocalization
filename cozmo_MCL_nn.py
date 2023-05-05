import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

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

# Arbitrary values, to model gaussian noise.
DATA_DIR = './model/data/0'
KIDNAP_DIR = './cozmo-images-kidnap'
KIDNAP_IMG_PATH = './cozmo-images-kidnap/kidnap.jpg'
img_name_list = [img_file for img_file in os.listdir(DATA_DIR) if img_file.endswith('jpg') and img_file[0].isdigit()]
degree_increment = 360.0 / len(img_name_list)
sensorVariance = 0.01
proportionalMotionVariance = 0.02

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
  ''' a, b: shape=(length, )
  '''
  return np.linalg.norm(a-b, axis=0)

def compare_embs(emb1, emb2):
  return euclidean_dist(emb1, emb2)

def compare_images(emb1, emb2):
  img1 = emb1/255.0
  img2 = emb2/255.0
  return ((img1 - img2)*(img1 - img2)).sum()

def measurement_model(pose_id, curr_id, img_list):
  # see Table 5.2, implementation of probability normal distribution
  diff = compare_images(img_list[f'{pose_id}.jpg'], img_list[f'{curr_id}.jpg'])
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (diff * diff) / (2 * sensorVariance))

def measurement_model_nn(pose_id, curr_emb, emb_list):
  # see Table 5.2, implementation of probability normal distribution
  emb_diff = compare_embs(emb_list[f'{pose_id}.jpg'], curr_emb)
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (emb_diff * emb_diff) / (2 * sensorVariance))

def motion_model(move_step, curr_pos):
  # making variance proportional to magnitude of motion command
  new_deg = curr_pos - move_step - sample_normal_distribution(abs(proportionalMotionVariance))*move_step
  return new_deg % 360

def sample_normal_distribution(variance):
  total = sum([(2.0 * random.random()) - 1.0 for _ in range(12)])
  return math.sqrt(variance) * total / 2.0

def MCL(robot: cozmo.robot.Robot):
  robot.camera.image_stream_enabled = True
  
  cfg = util_m.load_cfg('./model/config/configuration.json')
  # cfg['data_dir'] = './model/data'
  model = get_model(cfg)

  img_name_list = [img_file for img_file in os.listdir(DATA_DIR) if img_file.endswith('jpg') and img_file[0].isdigit()]
  # img_list = {img_name: cv2.imread(os.path.join(DATA_DIR, img_name), 0) for img_name in img_name_list}
  img_infer_list = {img_name: get_infer_img(os.path.join(DATA_DIR, img_name)) for img_name in img_name_list}
  emb_list = {img_name: get_emb(model, img_infer_list[img_name]) for img_name in img_infer_list.keys()}

  M = 200
  particles = np.random.randint(0, 360, (M,))

  TIME_STEP = 5
  for _ in range(TIME_STEP):
    # take new image
    util_r.rotate_robot(robot, 10, 'left')
    util_r.take_single_img(robot, KIDNAP_IMG_PATH)

    curr_emb = get_emb(model, get_infer_img(KIDNAP_IMG_PATH))
    
    # Initialize arrays to store poses, corresponding weights, and their normalized probabilities
    poses, weights = np.array([]), np.array([])
    for i in range(M):
      new_pose = motion_model(degree_increment, particles[i])
      # Get the source image (from initial rotation) to compare to by rounding to the nearest multiple of degree_increment
      rounded_pose = int(degree_increment * round(new_pose / degree_increment)) % 360
      pose_id = int(rounded_pose // degree_increment)
      weight = measurement_model_nn(pose_id, curr_emb, emb_list)
      # store this information
      poses = np.append(poses, rounded_pose)
      weights = np.append(weights, weight)

    probs = weights/np.sum(weights)
    cdf = np.sum(np.tril(probs), axis=1) 
    assert abs(cdf[-1] - 1.0) < 1e-8, 'last index in CDF must be 1.0'
    
    # Resample according to CDF
    new_particles = []
    for _ in range(M):
      p = random.random()
      idx = 0
      while p >= cdf[idx]: idx += 1
      new_particles.append(poses[idx])

    particles = np.array(new_particles)

    # visualize the robot's beliefs about it's current position
    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.hist(np.array(newParticles))
    # plt.show()

    # Sum up the belief probabilities, in 20 degree increments    
    step = 20
    prob_bins = [0 for i in range(0, 360, step)]
    for i in range(M):
      id = int(poses[i] // step)
      prob_bins[id] += probs[i]

    # Print an estimated position
    est_pos = np.argmax(prob_bins) * step
    print('est_position', est_pos)

    # based on the position the robot thinks it is in, rotate back to home
    util_r.rotate_robot(robot, est_pos, 'right')
    util_r.speak_robot(robot, "I am home")

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
  # MCL()
