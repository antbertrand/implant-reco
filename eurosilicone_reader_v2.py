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
__credits__ = ["Pierre-Julien Grizel"]
__license__ = "CLOSED SOURCE"
__version__ = "TBD"
__maintainer__ = "Pierre-Julien Grizel"
__email__ = "pjgrizel@numericube.com"
__status__ = "Production"

import logging
import os
import shutil
import time

import cv2

from step1_chip_detection import chip_detector
from step2_precise_circle import better_circle
from step3_angle_correction import orientation_fixer2
from step1_chip_detection import chip_detector
from step4_letter_detection import caracter_detector

abs_path = os.path.dirname(__file__)
print('1 =', abs_path)


def main():
    """Main program loop.
    """
    format_log = "%(asctime)s %(message)s"
    logging.basicConfig(
        filename="./activity.log", format=format_log, level=logging.DEBUG
    )

    # Start detectors
    ChipD = chip_detector.ChipDetector()
    OrienF = orientation_fixer2.OrientationFixer()
    CaracD = caracter_detector.CaracDetector()

    # Stetting up the useful paths
    input_path = os.path.join(abs_path, "tests/input/")
    full_path = os.path.join(abs_path, "tests/full/")
    chip1_path = os.path.join(abs_path, "tests/chip_step1/")
    chip2_path = os.path.join(abs_path, "tests/chip_step2/")
    chip3_path = os.path.join(abs_path, "tests/chip_step3/")
    chip4_path = os.path.join(abs_path, "tests/chip_step4/")

    # Here instead of being in the loop of the camera taking pictures,
    # we loop on a folder and process every image that is copied into it.
    while True:

        files = os.listdir(input_path)
        if files != []:
            im_name = files[0]
            # On the final program. This should go after the part where the CHIP image
            # is saved from the camera.
            # img_chip.save(output_fn)
            # But we will do the detection again with the RetinaNet. So we'll used
            # the FULL img.

            # Here we will open the image  from disk:
            # First, we move it to the full foler, meaning it's processed.
            shutil.move(input_path + im_name, full_path + im_name)
            img_full = cv2.imread(full_path + im_name, 0)


            # STEP 1
            start = time.time()
            # Detecting the chip
            box, score = ChipD.detect_chip(img_full)

            # Cropping the chip
            if box is None:
                print("The chip was not detected properly. Please try again !")
            elif score < 0.1:
                print("The confidence is too low on the classification, the chip \
                        detection will probably be false")
            else:
                chip_step1 = ChipD.crop_chip(img_full, box)

            end = time.time()
            print('Step1 inference time = ', end - start)

            # Saving it in chip_step1 folder (Temporary)
            cv2.imwrite(chip1_path + im_name, chip_step1)


            # STEP 2
            start = time.time()
            # Getting a better detection of the circle with Hough
            chip_step2 = better_circle.circle_finder(chip_step1)

            end = time.time()
            print('Step2 inference time = ', end - start)

            # Saving it in the chip_step2 folder (Temporary)
            cv2.imwrite(chip2_path + im_name, chip_step2)


            # STEP 3
            start = time.time()

            # Predicting angle
            predicted_orientation, chip_step3 = OrienF.classify_angle(chip_step2)

            end = time.time()
            print('Step3 inference time = ', end - start)

            # Saving it in the chip_step3 folder (Temporary)
            cv2.imwrite(chip3_path + im_name, chip_step3)

            # STEP 4
            start = time.time()

            # Detecting the caracters
            chip_step4, lines = CaracD.carac_detection(chip_step3)
            cv2.imwrite(chip4_path + im_name, chip_step4)
            print(lines)


            end = time.time()
            print('Step4 inference time = ', end - start)



            """
            # Get additional info
            best_angle = self.get_chip_angle(img_chip)
            if best_angle is not None:
                self.get_text_from_azure(img_chip, best_angle)

            # computeResults.saveImage(img_chip)
            print(
                "Image saved. Change/Turn prothesis. Waiting 5s before detecting again."
            )
            time.sleep(5)
            logging.info("Circle detected")

            # self.ui.displayImage(img)
            print("START READING...")
            """


if __name__ == '__main__':
    main()
