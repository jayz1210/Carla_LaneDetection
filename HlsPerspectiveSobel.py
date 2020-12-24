import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from PIL import Image
import PIL.ImageGrab as img


# Sobel operations:-absolute,magnitude,direction


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
    sxbinary = np.zeros_like(scaled_sobel)
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
    mbinary = np.zeros_like(scaled_magnitude)
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
    grad_binary = np.zeros_like(grad_dir)
    grad_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return grad_binary


# HLS conversion for colour gradient
def hls_select(img, thresh=(0, 255)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    L_channel = hls[:, :, 1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    binary_output = np.ones_like(L_channel).astype(np.float)
    binary_output[(L_channel > 60) & (L_channel <= 125)] = 0

    # binary_output = np.zeros_like(L_channel).astype(np.float)
    # binary_output[(s_channel > 10) & (s_channel <= 70)] = 1

    # binary_output = np.zeros_like(L_channel).astype(np.float)
    # binary_output[(s_channel > 10) & (s_channel <= 100)] = 1
    return binary_output

# Perspective Conversion

def perspective(image):
    # Locate points of the documents or object which you want to transform

    pts1 = np.float32([[355, 400], [450, 400], [image.shape[1], image.shape[0]], [250, image.shape[0]]])
    pts2 = np.float32([[350, 0], [500, 0], [500, image.shape[0]], [350, image.shape[0]]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (700,600 ), cv2.INTER_LINEAR)
    #cropped_result=cropImage(result)
    #return cropped_result
    return result

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

    return combined,result


image =cv2.imread('image/1dualpanel.jpg')

combined,actual = combinedHlsSobelPerspectiv(image, 9, thres=(40, 150), rad=(0.7, 1.3), thres_hls=(20, 105))
perpec = perspective(actual)
hls_frame = hls_select(image, thresh=(0,255))

cv2.imshow("perspective",perpec)
cv2.imshow('hls_frame', hls_frame)  # Inital Capture
cv2.imshow('sobel combine', combined)  # Transformed Capture
cv2.imshow('combined', actual)  # Transformed Capture
cv2.imshow('orignal Image', image)  # Transformed Capture
cv2.waitKey(0)
