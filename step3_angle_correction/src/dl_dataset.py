"""Script that will create the dataset with the 360 rotations of each image.
In this version has to be done for train, test and validation by changing paths. """


import os

import cv2
import imutils
import numpy as np

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
