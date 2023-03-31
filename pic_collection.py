import cozmo
from cozmo.util import degrees
import os
import cv2
import numpy as np
import img_processing as imgPr
from PIL import Image, ImageOps

IMG_DIR = 'cozmo-imgs'


def get_in_position(robot: cozmo.robot.Robot):
    robot.set_lift_height(0).wait_for_completed()
    robot.set_head_angle(degrees(15.0), in_parallel = True).wait_for_completed()

def take_an_img(robot: cozmo.robot.Robot, img_name):
    robot.camera.image_stream_enabled = True    # Enabling Cozmo Camera

    get_in_position(robot)

    img = robot.world.latest_image
    if img is not None:
        annotated = img.annotate_image()
        converted = annotated.convert()

        img = ImageOps.grayscale(converted)
        img = np.array(img)

        imgPr.save_img(img, img_name)
        print('IMG TAKEN')
    else:
        print('CANNOT TAKE IMG')

def take_imgs(robot: cozmo.robot.Robot, num_pic=15, img_dir=IMG_DIR):
    robot.camera.image_stream_enabled = True    # Enabling Cozmo Camera

    get_in_position(robot)
    
    currAngle = 0.0
    rotateAngle = 360.0/num_pic
    for i in range(num_pic):
        img = robot.world.latest_image
        if img is not None:
            annotated = img.annotate_image()
            converted = annotated.convert()
            
            img = ImageOps.grayscale(converted)
            img = np.array(img)

            if not os.path.exists(img_dir): os.makedirs(img_dir)
            imgPr.save_img(img, os.path.join(img_dir, f'{i}-{currAngle}.jpg'))
            
            robot.turn_in_place(degrees(rotateAngle), speed=degrees(20)).wait_for_completed()
            currAngle += rotateAngle
        else:
            get_in_position(robot)
            print('CANNOT TAKE IMG')

def collect_imgs(num_pic=20, img_dir='cozmo-imgs'):
    cozmo.run_program(lambda x : take_imgs(x, num_pic=num_pic, img_dir=img_dir))

def collect_an_img(img_name):
    cozmo.run_program(lambda x : take_an_img(x, img_name=img_name))

if __name__ == '__main__':
    cozmo.run_program(lambda x : take_imgs(x, num_pic=20, img_dir='cozmo-imgs-2'))
    