import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

import os, sys
import cv2
import pandas as pd     
import random
import numpy as np
from PIL import Image
import math 
import kidnap

import util
import util_robot as util_r

from cozmo_MCL_neuralnet import *

sensorVariance = 0.01
proportionalMotionVariance = 0.01
IMG_DIR = 'cozmo-images-kidnap/'

def main():
    panoPixelArray = cv2.imread("cozmo-images-kidnap/cropped-pano.jpg") #image to read in, should read in our pano (the cropped one)
    width = panoPixelArray.shape[1]
    
    # Initialize cozmo camera
    pixelWeights = [] # predictions

    M = 100   # Number of particles
    
    # Algorithm MCL Line 2
    # fill array with uniform values as starting predictions at initialize
    # starts as randomized particles, aka guesses for where the robot is facing
    particles = [random.randint(0, width) for _ in range(M)]

    # Saves preliminary predictions to a dataframe
    pointFrame = pd.DataFrame(particles, columns=['particles'])
    
    TIME_STEPS = 5
    for _ in range(TIME_STEPS):
        cv_cozmo_image2 = None

        kidnap_file = 'cozmo-images-kidnap/kidnap.jpg'
        util_r.take_single_img(robot, kidnap_file)
        cv_cozmo_image2 = util.crop_img(util.get_img_gray(kidnap_file), offset_w=0)
        
        # empty arrays that hold population number, and weight
        pixelPopulationNumber = []
        pixelWeights = []

        # Algorithm MCL Line 3
        for pose in particles:
        # Algorithm MCL Line 4
            newPose = motion_model(pose, width)
            # Algorithm MCL line 5:
            # map is [0, 1] interval space for movement, sensing distance from 0
            weight = measurement_model(cv_cozmo_image2, newPose) 
            
            # Algorithm MCL line 6:
            pixelWeights = np.append(pixelWeights,[weight])
            pixelPopulationNumber = np.append(pixelPopulationNumber,[newPose])

        # sum all weight, create new array size M, calculate probability
        probabilities = pixelWeights / pixelWeights.sum() 
        #Cumulative Distribution Function
        cdf = np.sum(np.tril(probabilities), axis=1) 

def motion_model(movement, currPose, width=sys.maxsize):
  # diff from Ben: dist(var)*movement instead of dist(var*movement)
  x = sample_normal_distribution(proportionalMotionVariance) * movement
  newPose = currPose - movement - x
  return max(0, min(width, newPose))

if __name__ == '__main__':
    # print(sample_normal_distribution(sensorVariance))

    movement, currPose = 5, 250
    for i in range(30):
        print(motion_model(movement, currPose))