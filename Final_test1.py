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

# IM_WIDTH = 640
# IM_HEIGHT = 480
IM_WIDTH = 800
IM_HEIGHT = 640

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
    #pts1 = np.float32([[280, 250], [370, 250], [0, IM_HEIGHT], [IM_WIDTH, IM_HEIGHT]])
    #pts1 = np.float32([[280, 250], [370, 250], [0, 480], [640, 480]])
    pts1 = np.float32([[310,385],[500,385],[0,580],[650,580]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    minv=cv2.getPerspectiveTransform(pts2,pts1)
    result = cv2.warpPerspective(resize, matrix, (500, 600))
    return resize,minv,result

# lane deetection
def find_lines(warped_img, nwindows=9, margin=30, minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_img[warped_img.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped_img.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window + 1) * window_height
        win_y_high = warped_img.shape[0] - window * window_height

        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img
    #return out_img


def fit_polynomial(binary_warped, nwindows=9, margin=50, minpix=50, show=True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img \
        = find_lines(binary_warped, nwindows=nwindows, margin=margin, minpix=minpix)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # # Plots the left and right polynomials on the lane lines
    # if show == True:
    #     plt.plot(left_fitx, ploty, color='yellow')
    #     plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, out_img     

def draw_lines(img, left_fit, right_fit, minv):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Find left and right points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix 
    unwarp_img = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]), flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    return cv2.addWeighted(img, 1, unwarp_img, 0.3, 0)

def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    
    left_fit_cr = np.polyfit(ploty*ymtr_per_pixel, left_fitx*xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty*ymtr_per_pixel, right_fitx*xmtr_per_pixel, 2)
    
    # find radii of curvature
    left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ymtr_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ymtr_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return (left_rad, right_rad)
    

def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ## Image mid horizontal position 
    #xmax = img.shape[1]*xmtr_per_pixel
    ymax = img.shape[0]*ymtr_per_pixel
    
    center = img.shape[1] / 2
    
    lineLeft = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    lineRight = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    
    mid = lineLeft + (lineRight - lineLeft)/2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0. :
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))
    
    return message    

def show_curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    dist_txt = dist_from_center(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    
    out_img = np.copy(img)
    avg_rad = round(np.mean([left_curvature, right_curvature]),0)
    cv2.putText(out_img, 'Average lane curvature: {:.2f} m'.format(avg_rad), 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(out_img, dist_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    return out_img    






def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    copy_img=np.copy(i3)

    xmtr_per_pixel=3.7/800
    ymtr_per_pixel=30/640

    Sobel_HLS_combine=combinedHlsSobelPerspectiv(copy_img, 9, thres=(40, 150), rad=(0.7, 1.3), thres_hls=(20, 105))
    resize,minv,perpec=perspective(Sobel_HLS_combine)

    left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(perpec, nwindows=12)
    lane_img = draw_lines(copy_img, left_fit, right_fit, minv)
    cv2.imshow("Sliding Window",lane_img)
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
  

    
    time.sleep(500)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')