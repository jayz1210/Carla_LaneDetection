import cv2
import numpy as np

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
    width = image.shape[1]

    # an array of polygon with only one polygon in it that is in form of a triangle
    polygons = np.array([[(0, height), (0, height // 2), (width, height // 2), (width, height)]])
    # creating a black image of same dimention of our image
    black_mask = np.zeros_like(image)
    # attaching the polygon to the black image with white boundaries of polygon
    cv2.fillPoly(black_mask, polygons, 255)
    # image obtained after and operation
    masked_image = cv2.bitwise_and(black_mask, image)
    return masked_image


def display_lines(image, lines):
    # appling that lines on a black mask image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # converting 2d image into 1d
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


image = cv2.imread('3dualwhite.jpg')

# copying the orignal image
lane_image = np.copy(image)

my_canny_image = canny(lane_image)
cropped_image = region_of_interest(my_canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
show_line_image = display_lines(lane_image, lines)
combined_image = cv2.addWeighted(lane_image, 0.8, show_line_image, 1, 1)

cv2.imshow('result', combined_image)
cv2.imshow('Cropped Canny', cropped_image)
cv2.waitKey(0)
