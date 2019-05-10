import json
import os
from PIL import Image
import cv2 as cv2
import numpy as np
import imutils
import random


IMAGES_PATH = '/storage/eurosilicone/img_corr_resized/'
OUTPUT_PATH = '/storage/eurosilicone/ds_rotated/'
images = os.listdir(IMAGES_PATH)

for index, im in enumerate(images):

    print(im)
    img = cv2.imread(IMAGES_PATH + im)
    img2 = cv2.resize(img, (224, 224))

    # Create test / train / valdiation directories
    dir_train = OUTPUT_PATH + 'train/'
    dir_test = OUTPUT_PATH + 'test/'
    dir_val = OUTPUT_PATH + 'val/'
    if not os.path.exists(dir_train):
            os.makedirs(dir_train)
    if not os.path.exists(dir_test):
            os.makedirs(dir_test)
    if not os.path.exists(dir_val):
            os.makedirs(dir_val)

    ndir = index%5


    if ndir >=0 and ndir <3:
        dir_final = dir_train
    if ndir ==3:
        dir_final = dir_val
    if ndir ==4:
        dir_final = dir_test


    for k in range(0,360):

        img3 = imutils.rotate(img2,k)

        output_dir = dir_final + str(k) + '/'
        print(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)




        cv2.imwrite( output_dir+im, img3 )

    print('image ', k, 'done')


    print('finito')
