#!/usr/bin/env python
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

IM_WIDTH = 640
IM_HEIGHT = 480

def canny(image):
    # coverting from rgb to single channel  of grey image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # reduce noise in image (gaussian filter)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny edge detection (for detecting edges of objects in image)
    cannyImage = cv2.Canny(blur, 50, 150)
    return cannyImage


def region_of_interest(image):
    height = image.shape[0]
    width =image.shape[1]
    # an array of polygon with only one polygon in it that is in form of a triangle
    polygons = np.array([[(0, height), (0, height//2),(width, height//2),(width, height)]])

    # creating a black image of same dimention of our image
    black_mask = np.zeros_like(image)
    # attaching the polygon to the black image with white boundaries of polygon
    cv2.fillPoly(black_mask, polygons, 255)
    # image obtained after and operation
    masked_image = cv2.bitwise_and(black_mask, image)
    return masked_image

def display_lines(image,lines):
    # appling that lines on a black mask image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # converting 2d image into 1d
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    # copying the orignal image

    lane_image = np.copy(i3)

    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, minLineLength=8 ,maxLineGap=20)
    line_image = display_lines(lane_image, lines)
    combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 0.2, 1)
    cv2.imshow('cropped',cropped_image)
    #cv2.imshow('result', combined_image)
   
    cv2.waitKey(1)

    #return i3


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0) 

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    
    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=1.9))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)
   
    sensor.listen(lambda data: process_img(data))
  

    
    time.sleep(50)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')