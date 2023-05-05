import cozmo
from cozmo.util import degrees

import random
import os
import numpy as np
from PIL import Image, ImageOps
import cv2
import cozmo_MCL
import cozmo_MCL_nn as mcl_nn

import util
import util_robot as util_r

IMG_DIR = 'cozmo-images-kidnap'

def kidnap_problem_solver(robot: cozmo.robot.Robot):
    # Spins the cozmo 360 degrees to get a panorama image of its current environment
    util_r.take_imgs(robot, num_pic=15, img_dir=IMG_DIR, is_stitch=False)
    
    # Turn robot a random amount to simulate a kidnapping & snap picture at new location
    util_r.random_rotate(robot)
        
    #Use MCL to find original position, takes images an tries to relocate
    mcl_nn.MCL(robot)



if __name__ == '__main__':
    #Test picture collection
    # cozmo.run_program(lambda x : picColl.take_imgs(x, num_pic=20, img_dir='cozmo-imgs-data1'))
    # cozmo.run_program(kidnap)
    # cozmo.run_program(takeSingleImage)

    

    ############ Run the kidnapped robot problem ###########################################
    cozmo.run_program(kidnap_problem_solver)
    # Generate histogram to display cozmo's beliefs on location
    util.makeHistogram()


