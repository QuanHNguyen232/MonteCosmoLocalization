'''
Modified work of: http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/#code
'''

#!/usr/bin/python
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#############
# Note: 
# orange is where Cozmo belives it is after localization
# blue is where Cozmo initially belives it is (before localization)
#############

# Used for making a histogram to display where cozmo thinks he is
def makeHistogram():
  print('creating Histogram ...')
  # Gets values from the csv file
  df = pd.read_csv("cozmo-images-kidnap/data.csv")
  # Gets width of graph from panorama
  pano = cv2.imread("cozmo-images-kidnap/c-Panorama.jpg") # our cropped panorama
  dimensions = pano.shape
  width = dimensions[1]

  originalPredictions = df['particles']
  newestPredictions = df['newParticles']

  # clf() clears the histogram. We found it compiles at the start, so whatever data is in 
  # the csv file when you run the code will be included in the final histogram with the new
  # data unless you clear it first. 
  plt.clf()
  fig, ax = plt.subplots()
  custom_bins = np.arange(0, width, 10) #create custom bins for histogram -> num predictions per range of 10
  ax.hist(originalPredictions,range = [0,width], bins = custom_bins)
  freq, bins, patches = ax.hist(newestPredictions,range = [0,width], bins = custom_bins) #newest belief predications
  ax.set_title('Cozmo MCL Predictions')
  ax.set_xlabel('Width of Panorama')
  ax.set_ylabel('Frequency of Predicitons')
  fig.savefig("cozmo-images-kidnap/hist.png")
  
  bin_max = np.where(freq == freq.max()) #returns an array, can be multiple maxs that are equal
  max_belief = bin_max[0] #get index where max bin is located in array
  max_belief = max_belief[0] * 10 #scale result up to be pixel location in pano
  
  return max_belief #return max_belief for use by MCL
  #print(max_belief)
#makeHistogram()