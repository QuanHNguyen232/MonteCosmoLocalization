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

def kidnap(robot: cozmo.robot.Robot):
    randomAngle = random.randint(30, 360)
    robot.turn_in_place(degrees(randomAngle), speed=degrees(20)).wait_for_completed()

def rotate_robot(robot: cozmo.robot.Robot, angle):
    robot.turn_in_place(degrees(angle), speed=degrees(20)).wait_for_completed()

def takeSingleImage(robot: cozmo.robot.Robot):
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
        imgPr.save_img(img, os.path.join(IMG_DIR, kidnap_img))
        print('IMG TAKEN')
    else:
        print('CANNOT TAKE IMG')

if __name__ == '__main__':
    IMG_DIR = 'cozmo-images-kidnap'
    kidnap_img = 'kidnapPhoto.jpg'
    NUM_IMGS = 20

    picColl.collect_imgs(NUM_IMGS, img_dir=IMG_DIR)
    cozmo.run_program(kidnap)
    cozmo.run_program(takeSingleImage)

    all_particles = []
    for i in range(NUM_IMGS):
        imgname = f'{IMG_DIR}/{i}-{i*(360.0/NUM_IMGS)}.jpg'
        all_particles.append(imgPr.normalize_img(imgPr.get_img(imgname)))
    
    
    n = len(all_particles)
    prob = np.ones(n) / n
    num_rotate = 3
    for i in range(num_rotate):
        img_name = f'{IMG_DIR}/currLoc.jpg'
        picColl.collect_an_img(img_name)
        measure = imgPr.normalize_img(imgPr.get_img(img_name))
        prob = my_MCL.MCLocalize(prob, all_particles, 1, measure)
        cozmo.run_program(lambda x : rotate_robot(x, angle=20))
    
    print([round(val, 3) for val in prob])


