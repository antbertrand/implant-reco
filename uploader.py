#!/usr/bin/env python
# encoding: utf-8
"""
uploader.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

Main dataset uploader (meant to be called with a cron)
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

import os
import time
import signal
import logging

#from azure.storage.file import FileService
from azure.storage.blob import BlockBlobService

# pylint: disable=F403
import settings

logger = logging.getLogger(__name__)

BLOB_CONNECTION_STRING = "BlobEndpoint=https://eurosilicone.blob.core.windows.net/;QueueEndpoint=https://eurosilicone.queue.core.windows.net/;FileEndpoint=https://eurosilicone.file.core.windows.net/;TableEndpoint=https://eurosilicone.table.core.windows.net/;SharedAccessSignature=sv=2018-03-28&ss=bfqt&srt=sco&sp=rwdlacup&se=2022-05-16T17:59:47Z&st=2019-05-16T09:59:47Z&spr=https&sig=svg3ojRIIKLE7%2Bje2e5Rz0TRibz5wasE75HmljLL67A%3D"
CONTAINER_NAME = "acquisitions"

def main():
    """Main runtime"""
    while True:
        # Look into Azure
        blob_service = BlockBlobService(
            connection_string=BLOB_CONNECTION_STRING
        )

        # Scan file, upload them one by one restlessly
        # Only upload root directory
        for root, dirs, files in os.walk(settings.ACQUISITIONS_PATH):
            for fn in files:
                logger.info("Uploading {}".format(fn))
                blob_service.create_blob_from_path(
                    "acquisitions", fn, os.path.join(root, fn)
                )
                os.remove(os.path.join(root, fn))

        # Pause before going at it again
        time.sleep(60)


# Main loop
if __name__ == "__main__":
    main()

