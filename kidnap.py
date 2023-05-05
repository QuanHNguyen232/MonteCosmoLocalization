import os
import cv2
import random
import numpy as np
from PIL import Image, ImageOps

import cozmo
from cozmo.util import degrees

import cozmo_MCL
import util
import util_robot as util_r


def kidnap_problem_solver(robot: cozmo.robot.Robot):
    # Spins the cozmo 360 degrees to get a panorama image of its current environment
    util_r.take_imgs(robot, num_pic=20, img_dir='cozmo-images-kidnap')
    # print('imgs collected')
    
    # Turn robot a random amount to simulate a kidnapping & snap picture at new location
    util_r.random_rotate(robot)
    # print('kidnap')
    
    #Use MCL to find original position, takes images an tries to relocate
    cozmo_MCL.MCL(robot)


if __name__ == '__main__':
    #Test picture collection
    # cozmo.run_program(lambda x : picColl.take_imgs(x, num_pic=20, img_dir='cozmo-imgs-data1'))
    # cozmo.run_program(kidnap)
    # cozmo.run_program(takeSingleImage)

    

    ############ Run the kidnapped robot problem ###########################################
    cozmo.run_program(kidnap_problem_solver)
    # Generate histogram to display cozmo's beliefs on location
    util.makeHistogram()


