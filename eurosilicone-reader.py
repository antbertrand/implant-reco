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
from PIL import Image, ImageTk
from threading import Thread
from PIL import Image
from settings import *
# from keyboard import Keyboard

#LOGGER = logging.getLogger(__name__)


class EurosiliconeReader(object):
    """Singleton class
    """

    def process_chip(self, img_chip):
        """Process chip
        """
        #Opencv to PILLOW image
        img_chip_pil = Image.fromarray(img_chip)
        img_chip_pil_rotate = img_chip_pil.copy()

        #init array results
        best_scores = np.array([0,0,0], dtype=float)
        best_boxes = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        best_deg = np.array([0,0,0], dtype=int)

        #check every image rotate, with 10 deg step
        for deg in range(0, 360, 10):
            #inference function
            is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_chip_pil_rotate)
            #Check the best score detection for each text
            if len(out_scores) == 3:
                for i in range(len(out_scores)):
                    if out_scores[i] > best_scores[i]:
                        best_scores[i] = out_scores[i]
                        best_deg[i] = deg

                        for y in range(len(out_boxes[i])):
                            best_boxes[i][y] = out_boxes[i][y]

                #rotate image before saving to visualize... (For dev)
                img_chip_pil_rotate = img_chip_pil.rotate(deg)
                open_cv_image = np.array(img_chip_pil_rotate)
                #computeResults.saveImage(open_cv_image)

            else :
                continue

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
                print("Unable to find circle. Move the object.")
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
            #self.process_chip(img_chip)

            # computeResults.saveImage(img_chip)
            print ("Image saved. Change/Turn prothesis. Waiting 5s before detecting again.")
            time.sleep(5)
            logging.info("Circle detected")

            # self.ui.displayImage(img)

if __name__ == '__main__':
    ec = EurosiliconeReader()
    ec.main()
