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
import glob
import re
from datetime import datetime as dt

from azure.storage.blob import BlockBlobService
import numpy as np
import cv2

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
                 model_path=abs_path + "/models/",  # Path where is stored the currently used model
                 container_name="weights",
                 model_connection_string="BlobEndpoint=https://eurosilicone.blob.core.windows.net/;QueueEndpoint=https://eurosilicone.queue.core.windows.net/;FileEndpoint=https://eurosilicone.file.core.windows.net/;TableEndpoint=https://eurosilicone.table.core.windows.net/;SharedAccessSignature=sv=2018-03-28&ss=bfqt&srt=sco&sp=rwdlacup&se=2022-05-16T17:59:47Z&st=2019-05-16T09:59:47Z&spr=https&sig=svg3ojRIIKLE7%2Bje2e5Rz0TRibz5wasE75HmljLL67A%3D",
                 ):

        download_it = False
        # Looking for the existing model
        model_names = glob.glob('{}retinanet_step1_resnet50_*'.format(model_path))


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
            container_name, prefix='retinanet_step1_resnet50_')
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

        self.model = models.load_model(model_path+newest_model_name, backbone_name='resnet50')
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
