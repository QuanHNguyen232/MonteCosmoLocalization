import cozmo
from cozmo.util import degrees
import os
import cv2
import numpy as np
import img_processing as imgP
from PIL import Image, ImageOps

IMG_DIR = 'cozmo-imgs'

def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey()   # press any key to close
    cv2.destroyAllWindows()

def save_img(img: np.ndarray, filepath: str):
    cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

def setup_robot(robot: cozmo.robot.Robot):
    robot.set_lift_height(0).wait_for_completed()
    robot.set_head_angle(degrees(15.0), in_parallel = True).wait_for_completed()

def take_img(robot: cozmo.robot.Robot, num_pic=10, img_dir=IMG_DIR):
    robot.camera.image_stream_enabled = True    # Enabling Cozmo Camera

    setup_robot(robot)
    
    currAngle = 0
    rotateAngle = 360.0/num_pic
    for i in range(num_pic):
        img = robot.world.latest_image
        annotated = img.annotate_image()
        converted = annotated.convert()
        
        img = ImageOps.grayscale(converted)
        img = np.array(img)

        if not os.path.exists(img_dir): os.makedirs(img_dir)
        save_img(img, os.path.join(img_dir, f'{i}-{currAngle}.jpg'))
        
        robot.turn_in_place(degrees(rotateAngle)).wait_for_completed()
        currAngle += rotateAngle

def collect_img(num_pic=20, img_dir='cozmo-imgs-2'):
    cozmo.run_program(lambda x : take_img(x, num_pic=num_pic, img_dir=img_dir))
    
if __name__ =='__main__':
    cozmo.run_program(lambda x : take_img(x, num_pic=20, img_dir='cozmo-imgs-2'))
    