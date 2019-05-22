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

import keras
from retinanet.keras_retinanet import models
from retinanet.keras_retinanet import models
from retinanet.keras_retinanet.utils.image import preprocess_image, resize_image

import numpy as np
import cv2

# Configure logging
logger = logging.getLogger(__name__)
logger.info("Starting Chip Detector module.")


class ChipDetector(object):

    """This class will detect the chip localization depending on our pre-trained model.
    """

    def __init__(self,
        model_path="./models/retinanet_detection_resnet50_inf.h5",):
        
        # Check if file exists
        if os.path.isfile(model_path):
            logger.info("Model found")
        else:
            logger.info("No model found. Please add the inference model in the models folder")


        # Load model
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.labels_to_names = {0: 'pastille'}


    def detect_chip(self, im):
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
        
        size = im.shape

        # Resize the image 
        WIDTH = 450
        HEIGHT = 300
        im2 = cv2.resize(im,(WIDTH,HEIGHT))
        scale1 = size[0]/300

        # Equalize histogram
        im2 = cv2.equalizeHist(im2)

        #Convert it back to rgb to enter the network
        im2 = cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB)

        # preprocess image for network
        im2 = preprocess_image(im2)
        im2, scale2 = resize_image(im2)

        # process image
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(im2, axis=0))

        # correct for image scale
        boxes *= scale1/scale2

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            logger.info("No model found. Please add the inference model in the models folder")
            # scores are sorted so we can break
            if score < 0.5:
                print("The confidence is too low on the bounding box prediction")
                break
            else:
                print("The confidence on the classification is of {}%".format(score*100))

            return box, score



def main():
    """Just a sample test of chip detection"""

    im_path = './dataset/img_test/'
    im_name = 'FULL-2019-04-26-160938.png'
    
    #Load the image in grayscale mode
    image = cv2.imread(im_path+im_name,0)
    
    # Measuring inference time
    detector = ChipDetector()
    start = time.time() 
    box, score = detector.detect_chip(image)
    end = time.time()
    print('Inference time = ',end - start)

    print('Predicted angle =', score, box)

main()