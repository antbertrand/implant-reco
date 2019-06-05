#!/usr/bin/env python
# encoding: utf-8
"""
orientation_fixer.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

This will fix chip orientation.
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

import os
import time
import logging
import glob
import re
from datetime import datetime as dt
from azure.storage.blob import BlockBlobService

from keras.models import load_model
import keras.backend as K
import numpy as np
import imutils
import cv2


from model_updater import update_model

# Configure logging
logger = logging.getLogger(__name__)
logger.info("Starting Orientation Fixer module.")

abs_path = os.path.dirname(__file__)

#pylint: disable=C0103


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    print(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


class OrientationFixer():
    """This class will fix image orientation depending on our pre-trained model.
    """

    def __init__(self,
                 model_prefix='rotnet_step3_resnet50_'):

        # Checking if the used model is the best
        model_path = update_model(abs_path, model_prefix)

        # Load model
        self.model = load_model(model_path, custom_objects={'angle_error': angle_error})

    def classify_angle(self, im):
        """Classify an image (np array or keras array)
        See here for difference:
        https://stackoverflow.com/questions/53718409/numpy-array-vs-img-to-array

        Parameters:
        im: nummpy array
        The image to classify

        Returns:
        angle : int
        in [0, 1, 2, : 360] corresponding to the angle in a counter clockwise direction.

        im_corr : np array
        The image with the orientation corrected
        """

        im_b = cv2.resize(im, (224, 224))

        im_b = cv2.cvtColor(im_b, cv2.COLOR_GRAY2RGB)

        im_b = im_b / 255
        im_b = np.expand_dims(im_b, axis=0)  # correct shape for classification
        predictions = self.model.predict(im_b)
        # taking index of the maximum %
        predictions = predictions.argmax(axis=1)
        angle = predictions[0]
        im_corr = imutils.rotate(im, -1 * angle)
        return angle, im_corr


def main():
    """Just a sample test of image rotation"""
    HEIGHT = 224
    WIDTH = 224

    model = load_model('../models/rotnet_chip_resnet50.hdf5',
                       custom_objects={'angle_error': angle_error})
    im_path = '../ds/ds_rotated/test_vrac/'
    im = 'FULL-2019-04-26-140952.png'

    image = cv2.imread(im_path + im)

    #
    # plt.imshow(image)
    # plt.show()
    #

    start = time.time()  # Measuring inference time
    fixer = OrientationFixer()
    predicted_orientation, im_corr = fixer.classify_angle(image)
    end = time.time()
    print('Inference time = ', end - start)

    print('Predicted angle =', predicted_orientation)
    # plt.imshow(im_corr)
    # plt.show()
