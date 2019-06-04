#!/usr/bin/env python
# encoding: utf-8
"""
complete_processing.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

The program gathering the different steps of processing.

Main features:
* Applies step 1 2 3 4 in a row.

"""
from __future__ import unicode_literals


import os
import time
import cv2

from step1_chip_detection import chip_detector
from step2_precise_circle import better_circle
from step3_angle_correction import orientation_fixer2
from step4_letter_detection import caracter_detector

abs_path = os.path.dirname(__file__)




class CompleteProcessor():

    """This class will apply all the steps of processing to an input image.
    """

    def __init__(self):

        # Start detectors
        self.ChipD = chip_detector.ChipDetector()
        self.OrienF = orientation_fixer2.OrientationFixer()
        self.CaracD = caracter_detector.CaracDetector()

        # Setting up some useful paths to save images
        self.input_path = os.path.join(abs_path, "tests/input/")
        self.full_path = os.path.join(abs_path, "tests/full/")
        self.chip1_path = os.path.join(abs_path, "tests/chip_step1/")
        self.chip2_path = os.path.join(abs_path, "tests/chip_step2/")
        self.chip3_path = os.path.join(abs_path, "tests/chip_step3/")
        self.chip4_path = os.path.join(abs_path, "tests/chip_step4/")



    def complete_process(self, img_full):
        """Apply the 4 steps of processing to a raw image ( out of the camera).

        Parameters:
        img_full: numpy array
        The img on which to do the processing


        Returns:
        text : str
        The detected text at the end.
        Should look like that "AB091   28/346CC  36CV"
        """


        im_name = "IMG-{}.png".format(time.strftime("%Y-%m-%d-%H%M%S"))

        # STEP 1
        start = time.time()
        # Detecting the chip
        box, score = self.ChipD.detect_chip(img_full)

        # Cropping the chip
        if box is None:
            print("The chip was not detected properly. Please try again !")
        elif score < 0.1:
            print("The confidence is too low on the classification, the chip \
                    detection will probably be false")
        else:
            chip_step1 = self.ChipD.crop_chip(img_full, box)

        end = time.time()
        print('Step1 inference time = ', end - start)

        # Saving it in chip_step1 folder (Temporary)
        #cv2.imwrite(self.chip1_path + im_name, chip_step1)


        # STEP 2
        start = time.time()
        # Getting a better detection of the circle with Hough
        chip_step2 = better_circle.circle_finder(chip_step1)

        end = time.time()
        print('Step2 inference time = ', end - start)

        # Saving it in the chip_step2 folder (Temporary)
        #cv2.imwrite(self.chip2_path + im_name, chip_step2)


        # STEP 3
        start = time.time()

        # Predicting angle
        predicted_orientation, chip_step3 = self.OrienF.classify_angle(chip_step2)

        end = time.time()
        print('Step3 inference time = ', end - start)

        # Saving it in the chip_step3 folder (Temporary)
        #cv2.imwrite(self.chip3_path + im_name, chip_step3)

        # STEP 4
        start = time.time()

        # Detecting the caracters
        chip_step4, lines = self.CaracD.carac_detection(chip_step3)

        # Converting lines ( list of str) to text ( str, with lines separated with /t)
        text = ''
        for line in lines:
            text += line + '/t'
        # Saving chip4 somewhere
        #cv2.imwrite(chip4_path + im_name, chip_step4)

        end = time.time()
        print('Step4 inference time = ', end - start)

        return text
