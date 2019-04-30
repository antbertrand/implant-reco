#!/usr/bin/env python
# encoding: utf-8
"""
eurosilicone-reader.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

The main EUROSILICONE program but without a GUI.

Main features:
* Opens and constantly reads camera
* If something is detected, print a line on the screen
* Writes the image in a quiet place
* Loop forever

"""
from __future__ import unicode_literals

__author__ = ""
__copyright__ = "Copyright 2016, NumeriCube"
__credits__ = ["Pierre-Julien Grizel", ]
__license__ = "CLOSED SOURCE"
__version__ = "TBD"
__maintainer__ = "Pierre-Julien Grizel"
__email__ = "pjgrizel@numericube.com"
__status__ = "Production"

import tempfile
import time
import os
import logging

import numpy as np
import camera
import detection_instance
import image
import cv2
import imutils
import yolo
import yolo_text
import yolo_char
from utils import detect_text, get_circles, microsoft_detection_text
from PIL import Image, ImageTk
from threading import Thread
from PIL import Image
from settings import *
# from keyboard import Keyboard

#LOGGER = logging.getLogger(__name__)


class EurosiliconeReader(object):
    """Singleton class
    """

    def get_chip_angle(self, img_chip):
        """Return chip angle OR None if no text has been found.
        """
        #Opencv to PILLOW image
        img_chip_pil = Image.fromarray(img_chip)
        img_chip_pil_rotate = img_chip_pil.copy()

        #init array results
        best_scores = np.array([0,0,0], dtype=float)
        best_boxes = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        best_deg = np.array([0,0,0], dtype=int)

        # Check every image rotate, with 10 deg step
        best_score = 0.0
        got_it = False
        for deg in range(0, 360, 10):
            # Rotate image
            img_chip_pil_rotate = img_chip_pil.rotate(deg)

            # Inference function. We want to detect THREE lines of text so we ignore if we have less than that.
            # Then we compute the best score and keep it
            is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_chip_pil_rotate)
            if len(out_scores) < 2:    # Ok, we authorize 2 lines
                continue
            got_it = True
            sum_scores = np.sum(out_scores)
            print("Deg {}: {}, {}".format(deg, sum_scores, self.text_detection.detect_image(img_chip_pil_rotate)))

            # Keep only the best angle
            if sum_scores > best_score:
                best_score = sum_scores
                best_deg = np.array([deg, deg, deg], dtype=int)

        # Display some information about what we detected
        if not got_it:
            print("NO TEXT DETECTED")
            return None
        print("BEST ANGLE: {}".format(best_deg))
        return best_deg[0]


    def get_text_from_azure(self, img_chip, angle):
        """From the given angle, automate reading from Azure
        """
        img_rot = imutils.rotate(img_chip, angle)
        image_data = cv2.imencode('.jpg', img_rot)[1].tostring()
        data = microsoft_detection_text(image_data)
        if not "recognitionResult" in data:
            print("NOTHING READ.")
            return
        for line in data['recognitionResult']['lines']:
            print(line['text'])
        return

        if len(best_scores) == 3:
            #Crop texts detection
            img_text1, img_text2, img_text3 = self.detect.get_text_area(best_boxes, best_deg[0], best_deg[1], best_deg[2])

            #Save texte images
            #computeResults.saveImage(img_text1)
            #computeResults.saveImage(img_text2)
            #computeResults.saveImage(img_text3)

            #Convert to PIL format
            img_text1_pil = Image.fromarray(img_text1)
            img_text2_pil = Image.fromarray(img_text2)
            img_text3_pil = Image.fromarray(img_text3)

            #############################################PART 3##########################################
            #Char detection
            #Init list of char
            list_line1 = []
            list_line2 = []
            list_line3 = []

            #inference function
            is_detected_t1, out_boxes_t1, out_scores_t1, out_classes_t1 = self.char_detection.detect_image(img_text1_pil)
            is_detected_t2, out_boxes_t2, out_scores_t2, out_classes_t2 = self.char_detection.detect_image(img_text2_pil)
            is_detected_t3, out_boxes_t3, out_scores_t3, out_classes_t3 = self.char_detection.detect_image(img_text3_pil)

            #if at least 1 detection by line...
            if (is_detected_t1 == True and is_detected_t2 == True and is_detected_t3 == True):
                for char in out_classes_t1:
                    list_line1.append(char)
                for char in out_classes_t2:
                    list_line2.append(char)
                for char in out_classes_t3:
                    list_line3.append(char)

                #convert from list to concatenated string
                list_line1_str = ''.join(map(str, list_line1))
                list_line2_str = ''.join(map(str, list_line2))
                list_line3_str = ''.join(map(str, list_line3))

                final_list = [list_line1_str, list_line2_str, list_line3_str]

                #Instructions for sending to Arduino and simulate keystrokes of a keyboard...
                #Send string by string
                self.keyboard.send(list_line1_str)
                self.keyboard.send("  ")
                self.keyboard.send(list_line2_str)
                self.keyboard.send("  ")
                self.keyboard.send(list_line3_str)

                #add Serial Number on the final picture
                #final_img = computeResults.addSerialNumber(final_list)
                logging.info("Serial Number written")
                #cam.saveImage(fullimg, img)
                #computeResults.saveImage(final_img)
                logging.info("Image saved")

            else :
                print ("Unable to find all char. Please move the prosthesis")
                logging.info("All char not found")
        else :
            print ("Unable to find all text area. Please move the prosthesis")
            logging.info("All texts not found")


    def main(self,):
        """Main program loop.
        """
        FORMAT = '%(asctime)s %(message)s'
        logging.basicConfig(filename='./activity.log', format=FORMAT, level=logging.DEBUG)
        self.cam = camera.Camera()
        #self.past_detection = yolo.YOLO()
        logging.info("Camera detected")
        #self.cam.saveConf()
        #logging.info("Camera configuration saved")
        self.cam.loadConf("acA5472-17um.pfs")
        logging.info("Camera configuration loaded")

        # Start detectors
        self.past_detection = yolo.YOLO()
        self.text_detection = yolo_text.YOLO()
        self.char_detection = yolo_char.YOLO()

        # Until death stikes, we read images continuously.
        print("START READING...")
        while True:
            # Grab image from camera
            fullimg, img = self.cam.grabbingImage()

            # Convert to PIL format
            img_pil = Image.fromarray(fullimg)
            # computeResults = image.Image(fullimg)

            # Pastille detection, save image on-the-fly.
            # The goal here is to perform a FIRST detection, wait for 1s and perform a SECOND detection
            # in order to avoid keeping/getting blurry images.
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)
            if not is_detected:
                continue

            print("DON'T MOVE!")
            time.sleep(1)
            fullimg, img = self.cam.grabbingImage()
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)

            if not is_detected:
                print("You moved...")
                continue

            # Get detected zone
            self.detect = detection_instance.DetectionInstance(fullimg)
            is_cropped, img_chip = self.detect.get_chip_area(out_boxes)

            # Save chip image
            output_fn = os.path.join(ACQUISITIONS_PATH, "CHIP-{}.png".format(time.strftime("%Y-%m-%d-%H%M%S")))
            Image.fromarray(img_chip).save(output_fn)
            output_fn = output_fn.replace("CHIP", "FULL")
            Image.fromarray(fullimg).save(output_fn)
            print ("Image saved: {}".format(output_fn))
            #img_chip.save(output_fn)

            # Get additional info
            best_angle = self.get_chip_angle(img_chip)
            if best_angle is not None:
                self.get_text_from_azure(img_chip, best_angle)

            # computeResults.saveImage(img_chip)
            print("Image saved. Change/Turn prothesis. Waiting 5s before detecting again.")
            time.sleep(5)
            logging.info("Circle detected")

            # self.ui.displayImage(img)
            print("START READING...")

if __name__ == '__main__':
    ec = EurosiliconeReader()
    ec.main()
