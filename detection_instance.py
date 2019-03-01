"""
class for detection instance
"""
import time
import requests
import http
import urllib.parse
import urllib
import math
import imutils
import cv2
import numpy as np

from utils import detect_text, get_circles, microsoft_detection_text

EAST_NET = cv2.dnn.readNet("./east.pb")

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

class DetectionInstance(object):
    """ A class to detect the implant chip and read the serial number
    """
    def __init__(self, frame, out_boxes):
        self.frame = frame
        self.chip_crop = (out_boxes[0][0], out_boxes[0][1], out_boxes[0][2], out_boxes[0][3])
        self.chip = None
        self.text_orientations = []
        self.orientation_used = None
        self.text = None
        self.network = EAST_NET

    def get_chip_area(self):
        """ Cropping and equalization of image
        """

        #img = imutils.resize(self.frame, height=600)
        #xratio = self.frame.shape[1]/float(img.shape[1])
        #yratio = self.frame.shape[0]/float(img.shape[0])

        # circles = get_circles(img, 1.1)
        # real_circle = None

        # if circles is None:
        #     return False

        # for circle in circles[0, :]:
        #     # if circle[2] > 120:
        #     #     continue
        #     real_circle = circle.copy()

        # if real_circle is None :
        #     return False

        # crop_coords = (
        #     max(0, real_circle[1]-self.chip_crop_size[0]//2)*yratio,
        #     min(img.shape[0], real_circle[1]+self.chip_crop_size[0]//2)*yratio,
        #     max(0, real_circle[0]-self.chip_crop_size[1]//2)*xratio,
        #     min(img.shape[1], real_circle[0]+self.chip_crop_size[1]//2)*xratio
        # )

        #image = self.frame[
         #   int(crop_coords[0]):int(crop_coords[1]),
         #   int(crop_coords[2]):int(crop_coords[3])
       # ]

        #img = imutils.resize(image, height=600)
        #xratio = image.shape[1]/img.shape[1]
        #yratio = image.shape[0]/img.shape[0]

        #img = improve_contrast(img)
        # for i in frange(0.8, 3.0, 0.1):
        #     circles = get_circles(img, i, maxr=200)
        #     #print('test: ', type(circles))
        #     if circles is not None and (circles > 0).any():
        #         break

        # if circles is not None:
        #     for circle in circles[0,:]:
        #         #if circle[2]>120:
        #         #    continue
        #         cv2.circle(img, (circle[0], circle[1]), circle[2], (0,255,0), 2)
        #         cv2.circle(img, (circle[0], circle[1]), 2, (0,0,255), 3)
        #         real_circle = circle.copy()

        # else:
        #     print("Cannot find circles")

        #cropSize = (400, 400)
        #cropCoords = (max(0, real_circle[1]-cropSize[0]//2)*yratio,min(img.shape[0], real_circle[1]+cropSize[0]//2)*yratio,
                      #max(0, real_circle[0]-cropSize[1]//2)*xratio,min(img.shape[1], real_circle[0]+cropSize[1]//2)*xratio)
        #print(cropCoords)
        #x_min,y_min,x_max,y_max
        # for cropping structure : ymin, ymax, xmin, xmax
        #image = self.frame[int(self.chip_crop[1]):int(self.chip_crop[3]), int(self.chip_crop[0]):int(self.chip_crop[2])]
        image = self.frame[int(self.chip_crop[0]):int(self.chip_crop[2]), int(self.chip_crop[1]):int(self.chip_crop[3])]

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.equalizeHist(image)

        self.chip = image

        return True, image

    def get_text_area(img, out_boxes):
        """ Crop for 3 text areas in the chip area
        """
        image1 = img[int(out_boxes[0][0]):int(out_boxes[0][2]), int(out_boxes[0][1]):int(out_boxes[0][3])]
        image2 = img[int(out_boxes[1][0]):int(out_boxes[1][2]), int(out_boxes[1][1]):int(out_boxes[1][3])]
        image3 = img[int(out_boxes[2][0]):int(out_boxes[2][2]), int(out_boxes[2][1]):int(out_boxes[2][3])]

        return image1, image2, image3


    def get_text_orientations(self):
        """ Search for 3 text areas in the chip area
        """
        image = cv2.cvtColor(self.chip, cv2.COLOR_GRAY2BGR)
        for deg in range(0, 360, 20):
            boxes = detect_text(image, self.network, deg)
            if len(boxes) > 2:
                print ("text detection ok")
                self.text_orientations.append(deg)


    def read_text(self):
        """ Extract the text with microsoft text recognition
        """
        text = []
        for degree in self.text_orientations:
            image = imutils.rotate(self.chip, degree)

            image_data = cv2.imencode('.jpg', image)[1].tostring()
            data = microsoft_detection_text(image_data)

            flag_aa = False
            if not 'recognitionResult' in data:
                print(data)
                continue
            for line in data['recognitionResult']['lines']:
                if (line['text'].startswith('AA')
                        and line['boundingBox'][0] < line['boundingBox'][2]):
                    flag_aa = True
                    text = []
                if flag_aa:
                    text.append(line['text'])
            if len(text) >= 3:
                self.text = text
                self.orientation_used = degree
                break
