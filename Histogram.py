'''
Credit belongs to http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/#code
'''

#!/usr/bin/python
import cv2
import pandas as pd
import random
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

#############
# Note: 
# orange is where Cozmo belives it is after localization
# blue is where Cozmo initially belives it is (before localization)
#############

# Used for making a histogram to display where cozmo thinks he is
def makeHistogram():
  # Gets values from the csv file
  df = pd.read_csv("data/data.csv")
  # Gets width of graph from panorama
  pano = cv2.imread("cozmo-images-kidnap\c-Panorama.jpg") # our cropped panorama
  dimensions = pano.shape
  width = dimensions[1]

  originalPredictions = df['particles']
  newestPredictions = df['newParticles']

  # clf() clears the histogram. We found it compiles at the start, so whatever data is in 
  # the csv file when you run the code will be included in the final histogram with the new
  # data unless you clear it first. 
  plt.clf()
  plt.hist(originalPredictions,range = [0,width], bins = width)
  plt.hist(newestPredictions,range = [0,width], bins = width)
  plt.title('Robot')
  plt.xlabel('Width of Panorama')
  plt.ylabel('Frequency of Predicitons')
  plt.savefig("hist.png")
#makeHistogram()