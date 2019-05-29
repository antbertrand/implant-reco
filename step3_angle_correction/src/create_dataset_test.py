
#!/usr/bin/env python
# encoding: utf-8
"""
This cript will download the dataset from azure. It will also create the rotated
dataset with the 360 rotations of each image.
In this version has to be done for train, test and validation by changing paths.
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
import hashlib
import base64

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
logger.info("Starting Dataset download module.")



DS_URL = "https://eurosilicone.blob.core.windows.net/dsorientation"


def download_dataset(output_path="/storage/eurosilicone/",
                     container_name="weights",
                     blob_name="rotnet_chip_resnet50.hdf5",
                     model_connection_string="BlobEndpoint=https://eurosilicone.blob.core.windows.net/;QueueEndpoint=https://eurosilicone.queue.core.windows.net/;FileEndpoint=https://eurosilicone.file.core.windows.net/;TableEndpoint=https://eurosilicone.table.core.windows.net/;SharedAccessSignature=sv=2018-03-28&ss=bfqt&srt=sco&sp=rwdlacup&se=2022-05-16T17:59:47Z&st=2019-05-16T09:59:47Z&spr=https&sig=svg3ojRIIKLE7%2Bje2e5Rz0TRibz5wasE75HmljLL67A%3D",
                     ):
    """Pre-load model if not already on disk
    """
    # Check if file exists / is fresh
    download_it = True
    if os.path.isfile(model_path):
        # File exists? Retreive online md5 from Azure
        target_blob_service = BlockBlobService(
            connection_string=model_connection_string
        )
        blob = target_blob_service.get_blob_properties(
            container_name=container_name,
            blob_name=blob_name,
        )
        blob_md5 = blob.properties.content_settings.content_md5

        # Read file md5 & compare
        file_md5 = self._read_file_md5(model_path)
        if file_md5 == blob_md5:
            download_it = False

    # Download if necessary
    if download_it:
        # Create target path if necessary
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)

        # Download
        logger.info("Downloading latest model")
        target_blob_service = BlockBlobService(connection_string=model_connection_string)
        target_blob_service.get_blob_to_path(
            container_name=container_name,
            blob_name=blob_name,
            file_path=model_path,
        )

    # Load model
    self.model = load_model(model_path, custom_objects={'angle_error': angle_error})



def main():
    """Just a sample test of image rotation"""
    HEIGHT = 224
    WIDTH = 224

    model = load_model('../models/rotnet_chip_resnet50.hdf5',custom_objects={'angle_error': angle_error})
    im_path = '../ds/ds_rotated/test_vrac/'
    im = 'FULL-2019-04-26-140952.png'

    image = cv2.imread(im_path+im)

    #
    # plt.imshow(image)
    # plt.show()
    #

    start = time.time() # Measuring inference time
    fixer = OrientationFixer()
    predicted_orientation, im_corr = fixer.classify_angle(image)
    end = time.time()
    print('Inference time = ',end - start)

    print('Predicted angle =', predicted_orientation)
    # plt.imshow(im_corr)
    # plt.show()

#IMAGES_PATH = "/storage/eurosilicone/ds_corr_resized/img_train/img_corr/"
#OUTPUT_PATH = "/storage/eurosilicone/ds_rotated/train/"
IMAGES_PATH = "/home/abert/Documents/NumeriCube/eurosilicone/gcaesthetics-implantbox/dataset/step3_orientationfixer/without_crop/img_test/img_corr/"
OUTPUT_PATH = "/home/abert/Documents/NumeriCube/eurosilicone/gcaesthetics-implantbox/dataset/step3_orientationfixer/without_crop/img_test/rotated/"
IMAGES = os.listdir(IMAGES_PATH)
SIDE = 224
RAYON = int(224/2)
mask = np.zeros((SIDE,SIDE), np.uint8)

# draw the outer circle
cv2.circle(mask,(RAYON,RAYON),RAYON,(255,255,255),-1)


for index, im in enumerate(IMAGES):

    print(im)
    img2 = cv2.imread(IMAGES_PATH + im, 0)

    for k in range(0, 360):
        # Creating rotation of the image by k degree
        img3 = imutils.rotate(img2, k)

        # k_3 is k written on 3 digits: 3 => 003
        k_3 = "{:03d}".format(k)

        # Creating folder k_3 if doesn't already exist
        output_dir = OUTPUT_PATH + str(k_3) + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Saves image in folder k_3
        masked_data = cv2.bitwise_and(img3, img3, mask=mask)
        cv2.imwrite( output_dir+im, masked_data )

    print("im", index, "done")


print("finito")
