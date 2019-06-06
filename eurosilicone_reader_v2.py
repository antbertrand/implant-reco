#!/usr/bin/env python
# encoding: utf-8
"""
eurosilicone-reader.py


The main EUROSILICONE program but taking the input images from a folder,
not the camera.

Main features:
* Loop forever on a folder and will process every image that is put into it.

"""
from __future__ import unicode_literals

__author__ = "Antoine Bertrand"
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

import numpy as np
import detection_instance
import imutils

from utils import detect_text, get_circles, microsoft_detection_text
from PIL import Image, ImageTk
from threading import Thread

from step1_chip_detection import chip_detector
from step2_precise_circle import better_circle
from step3_angle_correction import orientation_fixer
from step4_letter_detection import caracter_detector
import yolo_text

abs_path = os.path.dirname(__file__)


class EurosiliconeReader(object):
    """Singleton class to fully read labels
    """
    format_log = "%(asctime)s %(message)s"
    logging.basicConfig(
        filename="./activity.log", format=format_log, level=logging.DEBUG
    )


    def get_chip_angle(self, img_chip):
        """Return chip angle OR None if no text has been found.
        """
        # Opencv to PILLOW image
        img_chip_pil = Image.fromarray(img_chip)
        img_chip_pil_rotate = img_chip_pil.copy()

        # init array results
        best_scores = np.array([0, 0, 0], dtype=float)
        best_boxes = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        best_deg = np.array([0, 0, 0], dtype=int)

        # Check every image rotate, with 10 deg step
        best_score = 0.0
        got_it = False
        for deg in range(0, 360, 10):
            # Rotate image
            img_chip_pil_rotate = img_chip_pil.rotate(deg)

            # Inference function. We want to detect THREE lines of text so we ignore if we have less than that.
            # Then we compute the best score and keep it
            is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(
                img_chip_pil_rotate
            )
            if len(out_scores) < 2:  # Ok, we authorize 2 lines
                continue
            got_it = True
            sum_scores = np.sum(out_scores)
            print(
                "Deg {}: {}, {}".format(
                    deg,
                    sum_scores,
                    self.text_detection.detect_image(img_chip_pil_rotate),
                )
            )

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

    def simulate(self,):
        """Here instead of being in the loop of the camera taking pictures,
        we loop on a folder and process every image that is copied into it."""

        # Start detectors
        ChipD = chip_detector.ChipDetector()
        OrienF = orientation_fixer.OrientationFixer()
        CaracD = caracter_detector.CaracDetector()
        # self.text_detection = yolo_text.YOLO()

        # Stetting up the useful paths
        input_path = os.path.join(abs_path, "tests/input/")
        full_path = os.path.join(abs_path, "tests/full/")
        chip1_path = os.path.join(abs_path, "tests/chip_step1/")
        chip2_path = os.path.join(abs_path, "tests/chip_step2/")
        chip3_path = os.path.join(abs_path, "tests/chip_step3/")
        chip4_path = os.path.join(abs_path, "tests/chip_step4/")

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


                # STEP 1: we detect the chip
                start = time.time()
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
                cv2.imwrite(chip1_path + im_name, chip_step1)


                # STEP 2: we refine the chip
                # Getting a better detection of the circle with Hough
                start = time.time()
                chip_step2 = better_circle.circle_finder(chip_step1)
                end = time.time()
                print('Step2 inference time = ', end - start)
                cv2.imwrite(chip2_path + im_name, chip_step2)


                # STEP 3: we rotate the chip (predict chip angle)
                start = time.time()
                predicted_orientation, chip_step3 = OrienF.classify_angle(chip_step2)
                # angle = self.get_chip_angle(chip_step2)

                end = time.time()
                print('Step3 inference time = ', end - start)

                # Saving it in the chip_step3 folder (Temporary)
                cv2.imwrite(chip3_path + im_name, chip_step3)

                # STEP 4
                start = time.time()

                # Detecting the caracters
                chip_step4, lines = CaracD.carac_detection(chip_step3)
                cv2.imwrite(chip4_path + im_name, chip_step4)
                text = ''
                for line in lines:
                    text += line + '\t'

                end = time.time()
                print('Step4 inference time = ', end - start)

                print(text)




if __name__ == '__main__':
    ec = EurosiliconeReader()
    ec.simulate()
