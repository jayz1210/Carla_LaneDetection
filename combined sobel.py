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

def hls_select(img, thresh=(60, 125)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L_channel = hls[:, :, 1]

    binary_output = np.ones_like(L_channel).astype(np.float)
    binary_output[(L_channel > 60) & (L_channel <= 125)] = 0
    return binary_output

def abs_sobel_thresh(image, orient='x',sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # appliying soble
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # absolute value for sobelx
    abs_sobel = np.absolute(sobel)
    # convert absolute value into 8 bit
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel).astype(np.float)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    binary_output = np.copy(sxbinary)  # Remove this line
    return binary_output


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 3) Calculate the magnitude
    magnitude = np.sqrt(np.square(abs_sobelx) + np.square(abs_sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    mbinary = np.zeros_like(scaled_magnitude).astype(np.float)
    mbinary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return mbinary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    grad_binary = np.zeros_like(grad_dir).astype(np.float)
    grad_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return grad_binary 


# combining HLS and Sobel
def combinedHlsSobelPerspectiv(image,kSize,thres=(0,255),rad=(0,1),thres_hls=(60,125)):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=kSize, thresh=(thres[0],thres[1]))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=kSize, thresh=(thres[0],thres[1]))
    mag_binary = mag_thresh(image, sobel_kernel=kSize, mag_thresh=(thres[0],thres[1]))
    dir_binary = dir_threshold(image, sobel_kernel=kSize, thresh=(rad[0], rad[1]))

    combined = np.zeros_like(dir_binary).astype(np.float)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    hls_frame = hls_select(image, thresh=(thres_hls[0], thres_hls[1]))

    result = np.zeros_like(hls_frame).astype(np.float)
    result[((combined ==1) | (hls_frame == 1))] = 1

    return result    

# Perspective Conversion
 
def perspective(resize):
    pts1 = np.float32([[280, 250], [400, 250], [0, IM_HEIGHT], [IM_WIDTH, IM_HEIGHT]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(resize, matrix, (500, 600))
    return result 

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    copy_img=np.copy(i3)
    Sobel_HLS_combine=combinedHlsSobelPerspectiv(copy_img, 9, thres=(40, 150), rad=(0.7, 1.3), thres_hls=(20, 105))
    
    cv2.imshow("normal camera image", copy_img)
    cv2.imshow("Sobel and HLS combine ",Sobel_HLS_combine)
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