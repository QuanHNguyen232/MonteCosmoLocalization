import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageOps

import cozmo
from cozmo.util import degrees

import util

IMG_DIR = 'cozmo-images-kidnap'

def random_rotate(robot: cozmo.robot.Robot):
    img_list = [file for file in os.listdir('./cozmo-images-kidnap/') if (file[0].isdigit() and file.endswith('jpg'))]
    # except the last img since stitching does not work
    randomAngle = random.randint(30, 360 - (360/len(img_list)))
    rotate_robot(robot, randomAngle, 'left')

def rotate_robot(robot: cozmo.robot.Robot, angle:float, dir:str='left'):
    dir_val = 1 if dir=='left' else -1
    robot.turn_in_place(degrees(angle * dir_val), speed=degrees(45)).wait_for_completed()

def speak_robot(robot: cozmo.robot.Robot, msg):
    robot.say_text(msg).wait_for_completed()

def get_in_position(robot: cozmo.robot.Robot):
    robot.set_lift_height(0).wait_for_completed()
    robot.set_head_angle(degrees(15.0), in_parallel = True).wait_for_completed()

def take_single_img(robot: cozmo.robot.Robot, img_path:str=IMG_DIR):
    robot.camera.image_stream_enabled = True    # Enabling Cozmo Camera

    get_in_position(robot)

    img = robot.world.latest_image
    if img is not None:
        annotated = img.annotate_image()
        converted = annotated.convert()

        img = ImageOps.grayscale(converted)
        img = np.array(img)

        util.save_img(img, img_path)
        print('IMG TAKEN')
    else:
        print('CANNOT TAKE IMG')

def take_imgs(robot: cozmo.robot.Robot, num_pic:int=15, img_dir:str=IMG_DIR, is_stitch:bool=True):
    print('take_imgs ...')

    robot.camera.image_stream_enabled = True    # Enabling Cozmo Camera
    get_in_position(robot)
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    
    currAngle = 0.0
    rotateAngle = 360.0/num_pic
    for i in range(num_pic):
        img = robot.world.latest_image
        if img is not None:
            annotated = img.annotate_image()
            converted = annotated.convert()
            
            img = ImageOps.grayscale(converted)
            img = np.array(img)

            util.save_img(img, os.path.join(img_dir, f'{i}.jpg'))
            
            robot.turn_in_place(degrees(rotateAngle), speed=degrees(45)).wait_for_completed()
            currAngle += rotateAngle
        else:
            get_in_position(robot)
            print('CANNOT TAKE IMG')
    #stich panorama togehter from our images collected
    if is_stitch: util.stitching()

def collect_imgs(num_pic: int=20, img_dir: str='./cozmo-dataset'):
    cozmo.run_program(lambda x : take_imgs(x, num_pic, img_dir, False))

def collect_an_img(img_name: str):
    cozmo.run_program(lambda x : take_single_img(x, img_name))

