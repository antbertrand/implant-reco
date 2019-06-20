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
