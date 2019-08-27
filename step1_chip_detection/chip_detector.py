#!/usr/bin/env python
# encoding: utf-8
"""
orientation_fixer.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

This will detect the chip localization.
"""
from __future__ import unicode_literals

__author__ = ""
__copyright__ = "Copyright 2016, NumeriCube"
__credits__ = ["Pierre-Julien Grizel", ]
__license__ = "CLOSED SOURCE"
__version__ = "TBD"
__maintainer__ = "Pierre-Julien Grizel"
__email__ = "pjgrizel@numericube.com"
__status__ = "Developement"

import os
import time
import logging
import json

import numpy as np
import cv2

#from model_updater import update_model
from model_updater import check_model_md5

from .retinanet.keras_retinanet import models
from .retinanet.keras_retinanet.utils.image import preprocess_image, resize_image



# Configure logging
logger = logging.getLogger(__name__)
logger.info("Starting Chip Detector module.")

abs_path = os.path.dirname(__file__)


class ChipDetector():

    """This class will detect the chip localization depending on our pre-trained model.
    """

    def __init__(self,
                 model_name='retinanet_step1_resnet50_inf_20190605101500.h5'):

        # Checking if the used model is the best
        #model_path = update_model(abs_path, model_prefix)

        # Checking if the used model is the same as the one online
        model_path = check_model_md5(abs_path, model_name)
        print("MODEL1 = ", model_path)
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.labels_to_names = {0: 'pastille'}

    def detect_chip(self, im, im_name):
        """Detect a chip an image.

        Parameters:
        im: nummpy array
        The im on which to search for a chip

        Returns:
        box :
        Bounding box of the chip : [xmin ymin xmax ymax]

        score : float
        Classification score/confidence
        """

        size_im = im.shape

        # Resize the image
        WIDTH = 450
        HEIGHT = 300
        im2 = cv2.resize(im, (WIDTH, HEIGHT))
        scale1 = size_im[0] / 300

        # Equalize histogram
        im2 = cv2.equalizeHist(im2)

        # Convert it back to rgb to enter the network
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)

        # preprocess image for network
        im2 = preprocess_image(im2)
        im2, scale2 = resize_image(im2)

        # process image
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(im2, axis=0))
        if boxes is None or scores is None or labels is None:
            print("The chip was not detected properly. Please try again !")
            return None, None

        # correct for image scale
        boxes *= scale1 / scale2

        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            # Create supervisly labels from that prediction
            self.active_labeler(im_name, box)

            #print("The confidence on the classification is of {}%".format(score * 100))
            return box, score

    def crop_chip(self, im, box):
        """Crop the chip giving its bouding box.

        Parameters:
        im: nummpy array
        The im on which to crop the chip

        box: list [ xmin, ymin, xmax, ymax]
        The coordinates of the bouding box

        Returns:
        im_crop : numpy array
        The croppped chip
        """

        box = box.astype(int)
        im_crop = im[box[1]:box[3], box[0]:box[2]]

        return im_crop

    def active_labeler(self, im_name, box):
        """
        Creating labels from the predicitons that can be imported on supervisly
        and corrected if needed
        """
        null = None

        label_spvly = {"description": "", "tags": [],
                       "size": {"height": 3648, "width": 5472},
                       "objects": [],
                       }

        chip = {"description": "", "bitmap": null, "tags": [],
                "classTitle": "pastille",
                "points": {"exterior": [[], []], "interior": []}}

        # Changing labels
        chip["points"]["exterior"][0] = [box[0], box[1]]
        chip["points"]["exterior"][1] = [box[2], box[3]]


        label_spvly["objects"].append(chip)

        OUTPUT_PATH = "/home/numericube/Documents/current_projects/gcaesthetics-implantbox/tests/ann/step1/"
        with open(OUTPUT_PATH + im_name + '.json', 'w') as json_file:
            json.dump(label_spvly, json_file)


if __name__ == '__main__':
    """Just a sample test of chip detection"""

    im_path = '../dataset/img_test/'
    im_name = 'FULL-2019-04-26-160938.png'

    # Load the image in grayscale mode
    image = cv2.imread(im_path + im_name, 0)

    # Measuring inference time
    detector = ChipDetector()
    start = time.time()
    box_coord, score = detector.detect_chip(image)
    end = time.time()
    print('Inference time = ', end - start)

    print('Predicted angle =', score, box_coord)
