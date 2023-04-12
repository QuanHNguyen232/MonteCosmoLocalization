import img_processing as imgPr
import my_MCL
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

#This class performs and attempts to solve the kidnapped robot problem.
#(based on previous group's work)

def kidnap_problem_solver(robot: cozmo.robot.Robot):
    # Spins the cozmo 360 degrees to get a panorama image of its current environment
    picColl.collect_imgs(20, 'cozmo-imgs-testing')
    
    # Turn robot a random amount to simulate a kidnapping & snap picture at new location
    kidnap(robot)
    takeSingleImage(robot)
    
    #Use MCL to find original position
    cozmo_MCL.MCL(robot)
    
    #Generate histogram to display cozmo's beliefs on location
    Histogram.makeHistogram()


#Our work###############
#"Kidnap" robot by rotating a random amount
def kidnap(robot: cozmo.robot.Robot):
    randomAngle = random.randint(30, 360)
    robot.turn_in_place(degrees(randomAngle), speed=degrees(20)).wait_for_completed()

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
    
    # kidnap_img = 'kidnapPhoto.jpg'
    # NUM_IMGS = 20
    #
    # picColl.collect_imgs(NUM_IMGS, img_dir=IMG_DIR)
    # cozmo.run_program(kidnap)
    # cozmo.run_program(takeSingleImage)
    #
    # all_particles = []
    # for i in range(NUM_IMGS):
    #     imgname = f'{IMG_DIR}/{i}-{i*(360.0/NUM_IMGS)}.jpg'
    #     all_particles.append(imgPr.normalize_img(imgPr.get_img(imgname)))
    #
    # n = len(all_particles)
    # prob = np.ones(n) / n
    # num_rotate = 3
    # for i in range(num_rotate):
    #     img_name = f'{IMG_DIR}/currLoc.jpg'
    #     picColl.collect_an_img(img_name)
    #     measure = imgPr.normalize_img(imgPr.get_img(img_name))
    #     prob = my_MCL.MCLocalize(prob, all_particles, 1, measure)
    #     cozmo.run_program(lambda x : rotate_robot(x, angle=20))
    #
    # print([round(val, 3) for val in prob])
    

    #run the kidnapped robot problem
    cozmo.run_program(kidnap_problem_solver)



