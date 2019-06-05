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
# from keras.preprocessing import image
import numpy as np
import imutils
import cv2
# import matplotlib.pyplot as plt

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

    """This class will detect the chip localization depending on our pre-trained model.
    """

    def __init__(self,
                 model_path=abs_path + "/models/",  # Path where is stored the currently used model
                 container_name="weights",
                 model_connection_string="BlobEndpoint=https://eurosilicone.blob.core.windows.net/;QueueEndpoint=https://eurosilicone.queue.core.windows.net/;FileEndpoint=https://eurosilicone.file.core.windows.net/;TableEndpoint=https://eurosilicone.table.core.windows.net/;SharedAccessSignature=sv=2018-03-28&ss=bfqt&srt=sco&sp=rwdlacup&se=2022-05-16T17:59:47Z&st=2019-05-16T09:59:47Z&spr=https&sig=svg3ojRIIKLE7%2Bje2e5Rz0TRibz5wasE75HmljLL67A%3D",
                 ):

        download_it = False
        # Looking for the existing model
        model_names = glob.glob('{}rotnet_step3_resnet50_*'.format(model_path))

        if len(model_names) > 1:
            print(
                'TODO : Code something to keep only the newest model and remove the others')

        # Reading the date in the model's name
        else:
            model_name = os.path.basename(model_names[0])
            match = re.search(r'\d{4}\d{2}\d{2}\d{2}\d{2}\d{2}', model_name)
            used_model_date = dt.strptime(match.group(), '%Y%m%d%H%M%S')


        # Connection to the container on Azure
        blob_service = BlockBlobService(connection_string=model_connection_string)

        # List blobs in the container with a certain prefix
        generator = blob_service.list_blobs(
            container_name, prefix='rotnet_step3_resnet50_')
        newest_model_date = used_model_date
        newest_model_name = model_name
        # Compare each date in the models in azure with the currently used one
        for blob in generator:
            match = re.search(r'\d{4}\d{2}\d{2}\d{2}\d{2}\d{2}', blob.name)
            blob_model_date = dt.strptime(match.group(), '%Y%m%d%H%M%S')

            # If it's a newer one, we store it to later download it
            if blob_model_date > newest_model_date:
                newest_model_date = blob_model_date
                newest_model_name = blob.name
                download_it = True
                logger.info("Model found")

        # Download if necessary
        if download_it:
            # Download
            logger.info("Downloading latest model")
            target_blob_service = BlockBlobService(connection_string=model_connection_string)
            target_blob_service.get_blob_to_path(
                container_name=container_name,
                blob_name=newest_model_name,
                file_path=model_path+newest_model_name,
            )

        # Load model
        self.model = load_model(model_path+newest_model_name, custom_objects={
                                'angle_error': angle_error})



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
