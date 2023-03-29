import cozmo
from cozmo.util import degrees
import os
import cv2
import numpy as np
import imgprocessing as imgP
from PIL import Image, ImageOps

IMG_DIR = 'test-imgs'

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

        save_img(img, os.path.join(img_dir, f'{i}-{currAngle}.jpg'))
        
        robot.turn_in_place(degrees(rotateAngle)).wait_for_completed()
        currAngle += rotateAngle
        
    
if __name__ =='__main__':
    cozmo.run_program(take_img)
    