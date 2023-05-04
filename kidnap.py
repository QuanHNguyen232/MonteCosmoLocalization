import img_processing as imgPr
import MCL_old
import pic_collection as picColl
import random
import cozmo
from cozmo.util import degrees
import os
import numpy as np
from PIL import Image, ImageOps
import cv2
import Histogram
import cozmo_MCL


def kidnap_problem_solver(robot: cozmo.robot.Robot):
    # Spins the cozmo 360 degrees to get a panorama image of its current environment
    picColl.take_imgs(robot, num_pic=30, img_dir='cozmo-images-kidnap')
    # print('imgs collected')
    
    # Turn robot a random amount to simulate a kidnapping & snap picture at new location
    kidnap(robot)
    # print('kidnap')
    
    #Use MCL to find original position, takes images an tries to relocate
    cozmo_MCL.MCL(robot)
    
    # Generate histogram to display cozmo's beliefs on location
   # Histogram.makeHistogram()


#"Kidnap" robot by rotating a random amount
def kidnap(robot: cozmo.robot.Robot):
    randomAngle = random.randint(30, 360)
    robot.turn_in_place(degrees(randomAngle), speed=degrees(45)).wait_for_completed()

def rotate_robot(robot: cozmo.robot.Robot, angle):
    robot.turn_in_place(degrees(angle), speed=degrees(20)).wait_for_completed()

def takeSingleImage(robot: cozmo.robot.Robot):
    IMG_DIR = 'cozmo-images-kidnap' #Directory to save image to
    robot.camera.image_stream_enabled = True    # Enabling Cozmo Camera

    robot.set_lift_height(0).wait_for_completed()
    robot.set_head_angle(degrees(15.0), in_parallel = True).wait_for_completed()

    img = robot.world.latest_image
    if img is not None:
        annotated = img.annotate_image()
        converted = annotated.convert()

        img = ImageOps.grayscale(converted)
        img = np.array(img)

        if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
        imgPr.save_img(img, os.path.join(IMG_DIR, 'kidnapPhoto.jpg'))
        #print('IMG TAKEN')
    else:
        print('CANNOT TAKE IMG')


if __name__ == '__main__':
    #Test picture collection
    # cozmo.run_program(lambda x : picColl.take_imgs(x, num_pic=20, img_dir='cozmo-imgs-data1'))
    # cozmo.run_program(kidnap)
    # cozmo.run_program(takeSingleImage)

    

    ############ Run the kidnapped robot problem ###########################################
    cozmo.run_program(kidnap_problem_solver)
    # Generate histogram to display cozmo's beliefs on location
    Histogram.makeHistogram()


