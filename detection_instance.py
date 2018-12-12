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

class DetectionInstance(object):
    """ A class to detect the implant chip and read the serial number
    """
    def __init__(self, frame):
        self.frame = frame
        self.chip_crop_size = (200, 200)
        self.chip = None
        self.text_orientations = []
        self.orientation_used = None
        self.text = None
        self.network = EAST_NET

    def get_chip_area(self):
        """ Search for a circle to get the coordinates of the chip
        """
        img = imutils.resize(self.frame, height=600)
        xratio = self.frame.shape[1]/float(img.shape[1])
        yratio = self.frame.shape[0]/float(img.shape[0])

        circles = get_circles(img)

        for circle in circles[0, :]:
            if circle[2] > 120:
                continue
            real_circle = circle.copy()

        crop_coords = (
            max(0, real_circle[1]-self.chip_crop_size[0]//2)*yratio,
            min(img.shape[0], real_circle[1]+self.chip_crop_size[0]//2)*yratio,
            max(0, real_circle[0]-self.chip_crop_size[1]//2)*xratio,
            min(img.shape[1], real_circle[0]+self.chip_crop_size[1]//2)*xratio
        )

        image = self.frame[
            int(crop_coords[0]):int(crop_coords[1]),
            int(crop_coords[2]):int(crop_coords[3])
        ]

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)

        self.chip = image

        return True

    def get_text_orientations(self):
        """ Search for 3 text areas in the chip area
        """
        image = cv2.cvtColor(self.chip, cv2.COLOR_GRAY2BGR)
        for deg in range(0, 360, 20):
            boxes = detect_text(image, self.network, deg)
            if len(boxes) > 2:
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
            if len(text) == 3:
                self.text = text
                self.orientation_used = degree
                break
