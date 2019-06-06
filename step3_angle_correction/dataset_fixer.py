#!/usr/bin/env python
# encoding: utf-8
"""
dataset_fixer.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

Dummy file to fix supervisely's dataset shape
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


# Construct dataset from Supervisely's raw dataset export
import json
import os
import math
from shutil import copyfile

import numpy as np
import cv2
import imutils

def get_angle(ann):
    """Get angle from an annotation
    """
    # Calculate angle
    #Store the coordinates of the two points creating the vector
    x1 = int(ann['objects'][0]['points']['exterior'][0][0])
    y1 = int(ann['objects'][0]['points']['exterior'][0][1])
    x2 = int(ann['objects'][0]['points']['exterior'][1][0])
    y2 = int(ann['objects'][0]['points']['exterior'][1][1])

    size = np.zeros(2)
    size[0] = ann['size']['width']
    size[1] = ann['size']['height']
    print(size)
    # Removing the ".json" at the end to get the image name
    # im_name = filename
    # print(im_name)
    # img2= cv2.imread(IMAGE_PATH+im_name,0)

    #Check points role ( is it the center point or the other ?)
    # The one closer to the real center of the image is the center point
    xcenter = size[0]/2
    ycenter = size[1]/2
    dist1 = np.sqrt((ycenter-y1)**2+(xcenter-x1)**2)
    dist2 = np.sqrt((ycenter-y2)**2+(xcenter-x2)**2)
    print(dist1,dist2)

    if dist1 > dist2:
        xc = x2
        yc = y2
        x_bord = x1
        y_bord = y1

    else:
        xc = x1
        yc = y1
        xb = x2
        yb = y2
    print(xc,yc)

    # Get the angle
    a = xb - xc
    b = yb - yc
    c = np.sqrt((b)**2+(a)**2)
    cos_alpha = a/c
    sin_alpha = b/c
    print('cos_alpha =', cos_alpha)
    print('sin_alpha =', sin_alpha)

    #alpha = math.acos(cos_alpha)

    if sin_alpha>=0:
        alpha = -1*(math.acos(cos_alpha) + np.pi/2)
    else:
        alpha =(math.acos(cos_alpha) - np.pi/2)

    #plt.imshow(img2)
    #plt.show()

    # Petits chagements pour que ça colle dans ce cas là
    #alpha = -1*(alpha+np.pi/2)
    print(math.degrees(alpha))

    #img3 = rotateImage(img2,-1*math.degrees(alpha))

    alpha_correc = int(-1*math.degrees(alpha))
    if alpha_correc < 0:
        alpha_correc = 360 + alpha_correc
    return alpha_correc

# def generate_rotations():
#     for img_file in os.listdir(os.path.join(LOCAL_PATH, "ann")):
#         with open(os.path.join(LOCAL_PATH, "ann", json_file)) as j:
#             ann = json.load(j)
#             target = ann['tags'][0]['name']     # train / test
#             angle = get_angle(ann)
#             print(json_file, angle)




IMAGES_URL = "https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/tasks/archives/U/u/1m/YNeKPPE9LBLXCq4V26Eu3U58lnWCg2k8sW19fVhfWS3O7ZOqHsgfxqmnMjtxxJyADfNV6kG3puFyj76yllhA7n7zNouE0C6Tj5zVksnHJUPl2Z0gr5JLzlBzhnRm.tar"
LOCAL_PATH = "/Users/pjgrizel/Downloads/ES_orientation_chip-train-val/ds/"
for json_file in os.listdir(os.path.join(LOCAL_PATH, "ann")):
    with open(os.path.join(LOCAL_PATH, "ann", json_file)) as j:
        ann = json.load(j)
        target = ann['tags'][0]['name']     # train / test
        angle = get_angle(ann)
        print(json_file, angle)

        # Save somewhere else and we're good to go
        img_filename = json_file.replace(".json", "")
        img_path = os.path.join(LOCAL_PATH, "img", img_filename)
        dest_path = os.path.join(LOCAL_PATH, target, "%03d" % angle)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        img_dest_path = os.path.join(dest_path, "ROT%03d-%s" % (angle, img_filename))
        copyfile(img_path, img_dest_path)

        # Generate OTHER rotations for train set only
        if not target == "train":
            continue
        for gen_angle in range(1, 360):
            img = cv2.imread(img_dest_path)
            rotated = imutils.rotate(img, -gen_angle)
            res_angle = angle + gen_angle
            if res_angle >= 360:
                res_angle -= 360
            dest_path = os.path.join(LOCAL_PATH, target, "%03d" % res_angle)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            rotated_path = os.path.join(dest_path, "ROT%03d-%s" % (angle, img_filename))
            print(rotated_path)
            cv2.imwrite(rotated_path, rotated)
