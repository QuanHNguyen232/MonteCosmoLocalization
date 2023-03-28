import cozmo
from cozmo.util import degrees
import os
import sys
import time

#Directory to save files to
directory '.'


#Function to rotate and collect images to be converted into raw data
def cozmo_take_pictures(robot: cozmo.robot.Robot):
    #Set head angle and lift height before photo collection
    robot.set_head_angle(degrees(10.0)).wait_for_completed()
    robot.set_lift_height(0.0).wait_for_completed()
    #Make sure Cozmo is enabled to take pictures
    robot.camera.image_image_stream_enabled = True
    
    #Create directory for pictures
    if not os.path.exists('pictures'):
        os.makedirs('pictures')
    if not os.path.exists('pictures/'):
        os.makedirs('pictures/')

    #Loop to get pictures
    for i in range(20):
        robot.add_event_handler(cozmo.world.EvtNewCameraImage, picture_taken) #When new image taken, save it
        robot.turn_in_place(degrees(18).wait_for_completed())

def picture_taken(robot: cozmo.robot.Robot):
    pilImage = kwargs['image'].raw_image
    name = "-%d.jpg" % kwargs['image'].image_number
    pilImage.save(name, "JPEG")

#At very end, make sure that Cozmo runs the program
cozmo.run_program(cozmo_take_pictures)
