'''
Based on work of http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/#code
'''

#!/usr/bin/python
from string import hexdigits
from turtle import width
import cv2
import pandas as pd     
import random
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
from PIL import Image
import math 
import statistics as stat
import kidnap
import img_processing as imgPr
import Histogram

# Arbitrary values, to model gaussian noise.
sensorVariance = 0.01
proportionalMotionVariance = 0.01

def MCL(robot: cozmo.robot.Robot):
  panoPixelArray = cv2.imread("cozmo-images-kidnap/c-Panorama.jpg") #image to read in, should read in our pano (the cropped one)
  panoPixelArray.astype("float")                                    #make sure to change other references to desired image as needed in this file
  width = panoPixelArray.shape[1]
  # Initialize cozmo camera
  robot.camera.image_stream_enabled = True
  pixelWeights = [] # predictions

  M = 100   # Number of particles
  
  # Algorithm MCL Line 2
  # fill array with uniform values as starting predictions at initialize
  # starts as randomized particles, aka guesses for where the robot is facing
  particles = [random.randint(0, width) for _ in range(M)]
  '''
  Try this instead
  '''
  # particles = np.random.randint(0, width, (M, 1))

  # Saves preliminary predictions to a dataframe
  pointFrame = pd.DataFrame(particles, columns=['particles'])
  
  i = 0
  TIME_STEPS = 10
  while i < TIME_STEPS: # time steps is arbitrary    
    robot.turn_in_place(degrees(12.0)).wait_for_completed() #collect 30 images for pano, so 18 degrees per turn
    cv_cozmo_image2 = None

    kidnap.takeSingleImage(robot) 
    cv_cozmo_image2 = imgPr.get_img_gray("cozmo-images-kidnap/c-kidnapPhoto.jpg") #get cropped kidnapPhoto from prev method
    
    # empty arrays that hold population number, and weight
    pixelPopulationNumber = []
    pixelWeights = []

    # empty that can hold new population, temp array list for the one above
    newParticles = []

    # Algorithm MCL Line 3
    for pose in particles:
      # Algorithm MCL Line 4
      newPose = sample_motion_model(pose, width)
      # Algorithm MCL line 5:
      # map is [0, 1] interval space for movement, sensing distance from 0
      weight = measurement_model(cv_cozmo_image2, newPose) 
      
      # Algorithm MCL line 6:
      pixelWeights = np.append(pixelWeights,[weight])
      pixelPopulationNumber = np.append(pixelPopulationNumber,[newPose])

    # Compute probabilities (proportional to weights) and cumulative distribution function for sampling of next pose population
    # NOTE: This is the heart of weighted resampling that is _not_ given in the text pseudocode.
    # - first sum weights

    # sum all weight, create new array size M, calculate probability
    probabilities = pixelWeights / pixelWeights.sum() 
    #Cumulative Distribution Function
    cdf = np.sum(np.tril(probabilities), axis=1) 

   # redistribute population to newX

   #Algorithm MCL line 8:
   #resampling
    for m in range(M):
        p = random.uniform(0, 1)
        index = 0
        while p >= cdf[index]:
            index += 1
        newParticles.append(pixelPopulationNumber[index])
    particles = newParticles
    i += 1
  newParticles.sort()

  # updating the CSV file with the original predictions and the newest predictions
  df = pd.DataFrame(newParticles, columns = ['newParticles'])
  df = df.join(pointFrame) # joins new predictions with original predictions
  df = df.sort_values(by=['newParticles'], ascending=False)
  df.to_csv("cozmo-images-kidnap/data.csv", index = False)
  
  # Implement code to make Cozmo turn toward MCL's given highest belief probability:
  # - How to find max probability -> 'newParticles'
  # - Break up prob distribution into respective points that refer to degrees out of 360
  # - Mark beginning of pano as 'home' and uses this to find distance from believed location to 'home'
  # - Turn that num of degrees to 'home'
  
  #important: bin portions of data to find were most predictions are 'clumped,' then can take 
  #this bin and set as most believed location
  mostBelievedLoc = Histogram.makeHistogram()   # get max bin for 'newParticles' histogram, this is most frequent belief predication after MCL of a pixel range 10
                                                # (where Cozmo thinks it is after MCL)
  #get width of panorama, our 'map' of the environment
  '''
  I believe this part of reading image and get width is redundant since you did it on line 26
  '''
  pano = cv2.imread("cozmo-images-kidnap/c-Panorama.jpg") # our cropped panorama
  dimensions = pano.shape
  width = dimensions[1]
  
  #break up map of environment into pieces, corresponding to degrees out of 360
  widthToDegrees = width/360      # one degree out of 360 =  ___ of width
  
  # convert location of highest belief in map to degrees for Cozmo to turn
  # multiplied by small error percentage 0.95
  degreesToLocalize = 0.95*(mostBelievedLoc/widthToDegrees)

  #remove after done debugging
  print(f"Most believed location: {mostBelievedLoc}") #is series want to be float or int
  print(f"width to degrees: {widthToDegrees}")
  print(f"degrees to localize: {degreesToLocalize}")  #is series want to be float or int

  robot.turn_in_place(degrees(-degreesToLocalize))     #turn Cozmo accordingly (turn right to get Cozmo home -> MCL turns Cozmo left only to gather data)
  
   
  print("MCL Ran##############################################################")


def sample_motion_model(xPixel, width):
  # making variance proportional to magnitude of motion command
  newX = xPixel + sample_normal_distribution(abs(proportionalMotionVariance))
  return max(0, min(width, newX))

# map in this case is [0, 1] allowable poses
def measurement_model(latestImage, particlePose):
  # Gaussian (i.e. normal) error, see https://en.wikipedia.org/wiki/Normal_distribution
  # same as p_hit in Figure 6.2(a), but without bounds. Table 5.2
  img = Image.open("cozmo-images-kidnap/c-Panorama.jpg")
  width, height = img.size
  #get the slice of the panorama that corresponds to the pixel
  particle = slice(img, particlePose, 320, 320, height)
  particle = np.array(particle)
  #resize the images
  cv_particle = cv2.resize(particle, (width, height))
  image2 = cv2.resize(latestImage, (width, height))
  #compare how similar/different they are using MSE
  diff = compare_images(cv_particle, image2)
  #see Text Table 5.2, implementation of probability normal distribution
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (diff * diff) / (2 * sensorVariance))

def sample_sensor_model(pose):
  # ideal sensor will return pose (distance to right of 0), but we'll model Gaussian noise (could have more components as in class)
  return pose + sample_normal_distribution(sensorVariance)

#see Text Table 5.4, implementation of sample normal distribution
def sample_normal_distribution(variance):
  sum = 0
  for i in range(12):
    sum += (2.0 * random.random()) - 1.0
  return math.sqrt(variance) * sum / 2.0

# Compares images by pixels using Mean Squared Error formula
def compare_images(imageA, imageB):
  # See https://en.wikipedia.org/wiki/Mean_squared_error 
  dimensions = imageA.astype("float").shape
  width = dimensions[1]
  height = dimensions[0]
  '''
  Try this instead (lines 179-181)

  height, width = imageA.shape
  '''
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  # Dividing the values so they fit 
  err /= (width * height * width * height)
  return err

# Creates a slice of the panorama to compare to latestImage (kidnap photo)
def slice(img, center, pixelLeft, pixelRight, slice_size):
  # slice an image into parts slice_size wide
  # initialize boundaries
  img = img #Image.open("cozmo-images-kidnap\c-Panorama.jpg")
  # img = Image.open("cozmo-images-kidnap/c-Panorama.jpg")
  width, height = img.size
  ''' Line 197
  img.size returns width, height instead of height, width ???
  '''
  left = center - pixelLeft
  right = center + pixelRight

  # if we go out of bounds set the limit to be the bounds of the image
  if center < pixelLeft:
      left = 0
  if center > (width - pixelRight):
      right = width
  ''' Try this instead (line 193-200)
  
  left = max(center - pixelLeft, 0)
  right = min(center + pixelRight, width)
  '''
  # newImgSize = dim(center - 20, center + 20)
  upper = 0
  slices = int(math.ceil(width / slice_size))
  count = 1

  for slice in range(slices):
    # if we are at the end, set the lower bound to be the bottom of the image
    if count == slices:
      lower = width
    else:
      lower = int(count * slice_size)

      # box with boundaries
    bbox = (left, upper, right, lower)
    sliced_image = img.crop(bbox)
    cv_sliced = np.array(sliced_image)
    upper += slice_size
    # save the slice

    count += 1
    cv2.imwrite("cozmo-images-kidnap/sliced.jpg", cv_sliced)
    return sliced_image

# TO-DO

  #locate in panorama if it has gone a full 360
  #if pixels then knowing what is the point where things cycle around is important
  #subtract that number of pixels to wrap it around

  #slice
  #needs to wrap around if at one end or the other

  #figure out units for panorama so that motion model
  #(which should be wrapping around)
  #turning in degrees but units should be pixels
  #can't wrap around until we know...
  #what we are calling pixel 0

  #something that would print out where it thinks it is at 
  #and where it is facing as a demonstration of localization



if __name__ == '__main__':
  # robot = cozmo.robot.Robot
  # MCL(robot)
  
  #Testing MCL - Remove later#################################
  panoPixelArray = cv2.imread("cozmo-images-kidnap - Copy\Cropped.jpg") #image to read in, should read in our pano (the cropped one)
  panoPixelArray.astype("float")                                        #Make sure to change other references to desired image as needed in this file
  dimensions = panoPixelArray.shape
  width = dimensions[1]
  height = dimensions[0]
  # Initialize cozmo camera
  #robot.camera.image_stream_enabled = True
  pixelWeights = [] # predictions

