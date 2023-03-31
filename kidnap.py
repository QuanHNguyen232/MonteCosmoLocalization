import img_processing
import my_MCL
import pic_collection
import random
import cozmo
from cozmo.util import degrees
import os
import numpy as np
from PIL import Image, ImageOps
import cv2

def kidnap(robot: cozmo.robot.Robot):
    randomAngle = random.randint(30, 360)
    robot.turn_in_place(degrees(randomAngle)).wait_for_completed()


# def save_img(img: np.ndarray, filepath: str):
#     cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

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
        pic_collection.save_img(img, os.path.join(IMG_DIR, kidnap_img))
        print('IMG TAKEN')
    else:
        print('CANNOT TAKE IMG')

if __name__ == '__main__':
    IMG_DIR = 'cozmo-images-kidnap'
    kidnap_img = 'kidnapPhoto.jpg'

    pic_collection.collect_img(20, img_dir=IMG_DIR)
    cozmo.run_program(kidnap)
    cozmo.run_program(takeSingleImage)

    imgList = []
    for img_name in os.listdir(IMG_DIR):
        img = img_processing.get_img(f'{IMG_DIR}/{img_name}')
        img = img_processing.normalize_img(img)
        imgList.append(img)    

    measure = img_processing.get_img(f'{IMG_DIR}/{kidnap_img}')
    measure = img_processing.normalize_img(measure)
    # prob = my_MCL.MCLocalize(all_particles = imgList, move_step = 30, measurement = measure)
    # print(prob)


