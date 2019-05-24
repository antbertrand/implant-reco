"""Script that will create the dataset with the 360 rotations of each image.
In this version has to be done for train, test and validation by changing paths. """


import os

import cv2
import imutils





IMAGES_PATH = "/storage/eurosilicone/ds_corr_resized/img_train/img_corr/"
OUTPUT_PATH = "/storage/eurosilicone/ds_rotated/train/"
IMAGES = os.listdir(IMAGES_PATH)

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
        cv2.imwrite(output_dir + im, img3)

    print("im", index, "done")


print("finito")
