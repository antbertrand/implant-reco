"""
This module is used to get a better localization of the chip with
the Hough Circles method.
It also does some processing on the image before the next step by masking the
data outside the chip.
"""


import os
import logging

import numpy as np
import cv2




def circle_finder(img):
    '''
    Parameters 1 and 2 don't affect accuracy as such, more reliability.
    Param 1 will set the sensitivity; how strong the edges of the circles need to be.
    Too high and it won't detect anything, too low and it will find too much clutter.
    Param 2 will set how many edge points it needs to find to declare that it's found a circle.
    Again, too high will detect nothing, too low will declare anything to be a circle.
    The ideal value of param 2 will be related to the circumference of the circles.'''

    # 1. Preprocessing on the image
    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 11)

    # 2. Hough Circle method
    # Differenting different sizes of image so that parameters are
    # specifically chosen for a certain type.
    size = img.shape
    if size[0] > 1400 or size[1] > 1400:
        print('big chip')
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 10, 10000,
                                   param1=100, param2=100, minRadius=840, maxRadius=1000)
    else:
        print('other sizes of chips')
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 5, 10000,
                                   param1=100, param2=100, minRadius=100, maxRadius=600)

    # Taking only the first circle aka the one with the biggest certainty.
    c = circles[0, 0]

    # 3. Mask the image outside of the circle.
    mask = np.zeros(size, np.uint8)
    cv2.circle(mask, (c[0], c[1]), c[2], (255, 255, 255), -1)
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    # 4. Cropping the image at the circle's diameter size.
    # Assigning different the different dimensions
    c = c.astype(int)
    x_1, y_1, x_2, y_2 = c[0] - c[2], c[1] - c[2], c[0] + c[2], c[1] + c[2]

    # Conditions to handle cases where the circle goes outside the image
    if x_1 < 0:
        x_1 = 0
    if y_1 < 0:
        y_1 = 0
    if x_2 > size[1]:
        x_2 = size[1]
    if y_2 > size[0]:
        y_2 = size[0]

    # Cropping
    img_crop = img_masked[y_1:y_2, x_1:x_2]

    return img_crop
