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
        self.texte1 = None
        self.texte2 = None
        self.texte3 = None
        self.text_orientations = []
        self.orientation_used = None
        self.text = None
        self.network = EAST_NET

    def get_chip_area(self):
        """ Cropping and equalization of image
        """
        image = self.frame[int(self.chip_crop[0]):int(self.chip_crop[2]), int(self.chip_crop[1]):int(self.chip_crop[3])]

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)

        self.chip = image

        return True, image

    def get_text_area(self, out_boxes):
        """ Crop for 3 text areas in the chip area
        """
        self.texte1 = self.chip[int(out_boxes[0][0]):int(out_boxes[0][2]), int(out_boxes[0][1]):int(out_boxes[0][3])]
        self.texte2 = self.chip[int(out_boxes[1][0]):int(out_boxes[1][2]), int(out_boxes[1][1]):int(out_boxes[1][3])]
        self.texte3 = self.chip[int(out_boxes[2][0]):int(out_boxes[2][2]), int(out_boxes[2][1]):int(out_boxes[2][3])]


        return self.texte1, self.texte2, self.texte3


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
