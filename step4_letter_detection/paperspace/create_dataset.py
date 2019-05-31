import json
import os
from PIL import Image
import cv2 as cv2
import numpy as np
import imutils
import random


#Script the will create the dataset with the 360 rotations of each image

IMAGES_PATH = '/storage/eurosilicone/ds_corr_full/train/img_corr_full/'
OUTPUT_PATH = '/storage/eurosilicone/ds_rotated3/train/'
images = os.listdir(IMAGES_PATH)

SIDE = 224
RAYON = int(224/2)
mask = np.zeros((SIDE,SIDE), np.uint8)

# draw the outer circle
cv2.circle(mask,(RAYON,RAYON),RAYON,(255,255,255),-1)




for index, im in enumerate(images):

    print(im)
    img2 = cv2.imread(IMAGES_PATH + im, 0)
    #img2 = cv2.resize(img, (224, 224))

    # Create test / train / valdiation directories
    """
    dir_train = OUTPUT_PATH + 'train/'
    dir_test = OUTPUT_PATH + 'test/'
    dir_val = OUTPUT_PATH + 'val/'
    if not os.path.exists(dir_train):
            os.makedirs(dir_train)
    if not os.path.exists(dir_test):
            os.makedirs(dir_test)
    if not os.path.exists(dir_val):
            os.makedirs(dir_val)

    ndir = index%8


    if ndir >=0 and ndir <6:
        dir_final = dir_train
    if ndir ==6:
        dir_final = dir_val
    if ndir ==7:
        dir_final = dir_test

    """
    for k in range(0,360):

        img3 = imutils.rotate(img2,k)

        k_3 = "{:03d}".format(k)
        output_dir = OUTPUT_PATH + str(k_3) + '/'
        #print(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img4 = cv2.resize(img3, (224, 224))
        masked_data = cv2.bitwise_and(img4, img4, mask=mask)

        cv2.imwrite( output_dir+im, masked_data )


    print('im', index, 'done')


print('finito')
