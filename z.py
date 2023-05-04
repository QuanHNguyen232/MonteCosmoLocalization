import cozmo
from cozmo.util import degrees

import sys
import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt

degree_increment = 5
sensorVariance = proportionalMotionVariance = 0.01

def MCL(robot: cozmo.robot.Robot):
    robot.say_text("localizing").wait_for_completed()
    robot.camera.image_stream_enabled = True

    # generate a (random) initial population of M possible positions
    M = 150
    particles = [] 
    for i in range(M):
        particles.append(random.randint(0, 360))

    poses_and_weights = None

    # iterate until any given 20 degree bin contains 40% of the robot's belief likelihood.
    max_prob_bin = 0
    est_position = None
    while max_prob_bin < 0.4:
        # take pictures in 10 degree increments
        robot.turn_in_place(degrees(-10.0)).wait_for_completed()
        latest_image = robot.world.latest_image
        while latest_image is None:
            latest_image = robot.world.latest_image
        annotated = latest_image.annotate_image()
        if latest_image is not None:
            converted = annotated.convert()
            converted.save("latestImage.jpeg", "JPEG", resolution=10)

        # Initialize arrays to store poses, corresponding weights, and their normalized probabilities
        poses_and_weights = np.empty([M, 3])
        newPosition = None
        # for each potential position
        for p in range(M):
            currentPosition = particles[p]
            # update our belief about where the given pose represents, given the movement just made
            newPosition = motion_model(degree_increment, currentPosition)
            # Assign a weight to this position based on the image difference
            weight = measurement_model(newPosition) 
            # store this information
            poses_and_weights[p] = [newPosition, weight, 0]

        # normalize the relative likelihoods
        total_weight = 0.0
        for p in range(M):
            total_weight += poses_and_weights[p, 1]
        # store the normalized probabilities
        for p in range(M): 
            poses_and_weights[p,2] = poses_and_weights[p, 1] / total_weight
        # make CDF
        sum = 0
        cdf = []
        for p in range(M):
            sum += poses_and_weights[p,2]
            cdf.append(sum)
        cdf[M-1] = 1.0

        # Resample, according to this CDF
        newParticles = []
        for p in range(M):
            p = random.random()
            index = 0
            while p >= cdf[index]:
                index += 1
            newParticles.append(poses_and_weights[index,0])
        # Specify the new population of positions for the next iteration
        particles = newParticles

        # visualize the robot's beliefs about it's current position
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.hist(np.array(newParticles))
        plt.show()
        # Sum up the belief probabilities, in 20 degree increments
        bin_width = 20
        prob_bins = [0 for i in range(0, 360, bin_width)]
        for (pose, weight, prob) in poses_and_weights:
            prob_bins[int(pose // bin_width)] += prob
        # Print an estimated positoin
        if max_prob_bin != 0:
            est_position = prob_bins.index(max(prob_bins)) * bin_width
            print(f'est: {est_position}')
        print(f'The 20 degree bin with the higher probability has a probability of {max(prob_bins)}')
        # update the probability in the max bin so the robot can continue
        # if it is still unsure
        max_prob_bin = max(prob_bins)

    # based on the position the robot thinks it is in, rotate back to home
    robot.turn_in_place(degrees(-est_position)).wait_for_completed()
    robot.say_text("I'm hooooooooome!").wait_for_completed()

def measurement_model(particlePose):
  # Get the source image (from initial rotation) to compare to by rounding to the nearest multiple
  # of degree_increment
  roundedPose = int(degree_increment * round(float(particlePose) / degree_increment)) % 360 # assume degree_increment = 5
  # Fetch the source image from this position
  source_image_path = os.path.join(path, 'image' + str(roundedPose) + ".jpeg")
  # Read in the source image and the latest
  source_image = cv2.imread(source_image_path)
  latest_image = cv2.imread('latestImage.jpeg')

  minDiff = sys.maxsize
  # Sliding window:
  # We initialize the difference as the max possible ...
  height, width = source_image.shape[0:2]
  # Each offset represents a different sliding image pair 
  for offset in range(90, 10, -10):
    # crop the images, getting the overlapping portions
    overlap = width - offset
    source_cropped = source_image[0:height, 0:overlap]
    latest_cropped = latest_image[0:height, offset:width]
    # compare the cropped images
    diff = compare_images(source_cropped, latest_cropped)
    # update minDiff. We assume that the least image difference should be used for each of these
    # according to the fact that even slightly offset images may report large differences
    # even if they are very similar
    minDiff = min(minDiff, diff)
  
  # reverse the roles of the images for cases where the latest image spills over the right hand
  # side of the source image
  for offset in range(10, 90, 10):
    overlap = width - offset
    source_cropped = source_image[0:height, offset:width]
    latest_cropped = latest_image[0:height, 0:overlap]
    diff = compare_images(source_cropped, latest_cropped)
    minDiff = min(minDiff, diff)

  # the least difference should be used for the differences of these two images
  diff = minDiff
  #see Text Table 5.2, implementation of probability normal distribution
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (diff * diff) / (2 * sensorVariance))

def motion_model(movement, current_position):
  # making variance proportional to magnitude of motion command
  newDeg = current_position - movement - sample_normal_distribution(abs(movement * proportionalMotionVariance)) 
  # apply modulus to make sure the newX wraps around when it passes over either edge of the panorama
  return newDeg % 360

def sample_normal_distribution(variance):
  sum = 0
  for i in range(12):
    sum += (2.0 * random.random()) - 1.0
  return math.sqrt(variance) * sum / 2.0

if __name__ == '__main__':
    M = 100
    particles = [random.randint(0, 360) for i in range(M)] 
    # print(particles)

    poses_and_weights = None

    # iterate until any given 20 degree bin contains 40% of the robot's belief likelihood.
    max_prob_bin = 0
    est_position = None
    poses_and_weights = []
    while max_prob_bin < 0.4:
        for p in range(M):
            currentPosition = particles[p]
            # update our belief about where the given pose represents, given the movement just made
            newPosition = motion_model(degree_increment, currentPosition)
            
            particlePose = newPosition
            print(f'currentPosition={currentPosition} \t newPosition={round(newPosition, 5)} \t roundedPose={int(degree_increment * round(float(particlePose) / degree_increment)) % 360} \t {round(float(particlePose) / degree_increment)}')
            # Assign a weight to this position based on the image difference
            # weight = measurement_model(newPosition)
            # store this information
            # poses_and_weights.append([newPosition, weight, 0])
        break