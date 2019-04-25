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
import camera
import detection_instance
import image
import cv2
import os
import imutils
import time
import logging
import yolo
import yolo_text
import yolo_char
from PIL import Image, ImageTk
from threading import Thread
from PIL import Image
# from keyboard import Keyboard

#LOGGER = logging.getLogger(__name__)


class EurosiliconeReader(object):
    """Singleton class
    """
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

            # Pastille detection, save image on-the-fly
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)
            

            # Text Detection
            if is_detected:
                detect = detection_instance.DetectionInstance(fullimg)
                is_cropped, img_chip = detect.get_chip_area(out_boxes)
                # computeResults.saveImage(img_chip)
                print ("Image saved. Change/Turn prothesis")
                logging.info("Circle detected")
                print ("Image saved")
            else:
                print ("Unable to find circle. Please move the prosthesis")
                logging.info("Circle not found")
                with tempfile.NamedTemporaryFile(suffix=".png") as fp:
                    img_pil.save(fp.name)
                    os.system("img2txt -f ansi -W 100 {}".format(fp.name))

            # self.ui.displayImage(img)

if __name__ == '__main__':
    ec = EurosiliconeReader()
    ec.main()
