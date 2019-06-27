#!/usr/bin/env python
# encoding: utf-8
"""
model_updater.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

This will download a newer model if it's found on azure.
"""


import glob
import os
import logging
import re
from datetime import datetime as dt
from azure.storage.blob import BlockBlobService
import hashlib
import base64

# Configure logging
logger = logging.getLogger(__name__)


def update_model(abs_path,
                 model_type,
                 container_name="weights",
                 model_connection_string="BlobEndpoint=https://eurosilicone.blob.core.windows.net/;QueueEndpoint=https://eurosilicone.queue.core.windows.net/;FileEndpoint=https://eurosilicone.file.core.windows.net/;TableEndpoint=https://eurosilicone.table.core.windows.net/;SharedAccessSignature=sv=2018-03-28&ss=bfqt&srt=sco&sp=rwdlacup&se=2022-05-16T17:59:47Z&st=2019-05-16T09:59:47Z&spr=https&sig=svg3ojRIIKLE7%2Bje2e5Rz0TRibz5wasE75HmljLL67A%3D",
                 ):
    """Compares the date of the models stored on azure to the one stored locally.

    Parameters:
    model_type: str
    The type of the model, it will be used as a prefix to lok for the correct modelself.
    Example : "retinanet_step4_resnet50_inf_"

    abs_path: str
    The path from which this function has been called. Use to find the location of the modelsself.

    Returns:
    model_path : str
    The complete path leading to the newest model.

    """

    # Path where is stored the currently used model
    model_path = abs_path + "/models/"
    print('PATH============', model_path)
    download_it = False

    # Looking for the existing model
    model_names = glob.glob('{}{}*'.format(model_path, model_type))
    print(model_names, len(model_names))
    if len(model_names) > 1:
        print(
            'TODO : Code something to keep only the newest model and remove the others')

    # Reading the date in the model's name
    elif len(model_names) == 1:
        # Taking only the name, not the full path
        model_name = os.path.basename(model_names[0])
        match = re.search(r'\d{4}\d{2}\d{2}\d{2}\d{2}\d{2}', model_name)
        used_model_date = dt.strptime(match.group(), '%Y%m%d%H%M%S')

    # If there is not model stored locally
    elif model_names == []:
        used_model_date = dt.min
        model_name = None

    # Connection to the container on Azure
    blob_service = BlockBlobService(connection_string=model_connection_string)

    # List blobs in the container with a certain prefix
    generator = blob_service.list_blobs(
        container_name, prefix=model_type)
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
        target_blob_service = BlockBlobService(
            connection_string=model_connection_string)
        target_blob_service.get_blob_to_path(
            container_name=container_name,
            blob_name=newest_model_name,
            file_path=model_path + newest_model_name,
        )

    return model_path + newest_model_name

def _read_file_md5(fname):
    """Read md5 from file
    Taken from: https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as fcontent:
        for chunk in iter(lambda: fcontent.read(4096), b""):
            hash_md5.update(chunk)
    return base64.b64encode(hash_md5.digest()).decode("ascii")


def check_model_md5(abs_path,
                    blob_name,
                    container_name="weights",
                    model_connection_string="BlobEndpoint=https://eurosilicone.blob.core.windows.net/;QueueEndpoint=https://eurosilicone.queue.core.windows.net/;FileEndpoint=https://eurosilicone.file.core.windows.net/;TableEndpoint=https://eurosilicone.table.core.windows.net/;SharedAccessSignature=sv=2018-03-28&ss=bfqt&srt=sco&sp=rwdlacup&se=2022-05-16T17:59:47Z&st=2019-05-16T09:59:47Z&spr=https&sig=svg3ojRIIKLE7%2Bje2e5Rz0TRibz5wasE75HmljLL67A%3D",
                    ):
    """Pre-load model if not already on disk
    """
    # Path where is stored the currently used model
    model_path = os.path.join(abs_path, "models", blob_name)
    print(model_path)

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
        file_md5 = _read_file_md5(model_path)
        if file_md5 == blob_md5:
            download_it = False

    # Download if necessary
    if download_it:
        # Create target path if necessary
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)

        # Download
        logger.info("Downloading latest model")
        target_blob_service = BlockBlobService(
            connection_string=model_connection_string)
        target_blob_service.get_blob_to_path(
            container_name=container_name,
            blob_name=blob_name,
            file_path=model_path,
        )
    return os.path.join(model_path, blob_name)
