#!/usr/bin/env python
# encoding: utf-8
"""
orientation_fixer.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

This will detect the chip localization.
"""
from __future__ import unicode_literals


import os
import time
import logging

import cv2
import numpy as np

import keras

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

from model_updater import update_model
from .retinanet.keras_retinanet import models
from .retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from .retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from .retinanet.keras_retinanet.utils.colors import label_color



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# Configure logging
logger = logging.getLogger(__name__)
logger.info("Starting Chip Detector module.")

abs_path = os.path.dirname(__file__)


class CaracDetector():

    """This class will detect the caracters on a chip.
    """

    def __init__(self,
                 model_prefix='retinanet_step4_resnet50_inf'):

        # Checking if the used model is the best
        model_path = update_model(abs_path, model_prefix)

        # Load model
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.labels_to_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                                6: '6', 7: '7', 8: '8', 9: '9', 10: '/', 11: 'A', 12: 'B', 13: 'C',
                                14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K',
                                22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R',
                                29:'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'}


    def draw_caption(self, image, box, caption):
        """ Draws a caption above the box in an image.

        # Arguments
            image   : The image to draw on.
            box     : A list of 4 elements (x1, y1, x2, y2).
            caption : String containing the text to draw.
        """
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 4, (252, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 255), 1)

    def carac_detection(self, im):
        """Detects the caracters on an image.

        Parameters:
        im: nummpy array
        The im on which to search the caracters

        Returns:
        boxes : list of boxes
        Bounding boxes of all the detected caracters: [xmin ymin xmax ymax]

        scores : list of float
        Classification score/confidence

        labels : list of int
        Label number. Convert it to the label name with labels_to_names
        """


        # Resizing the image
        WIDTH = 800
        HEIGHT = 800
        image = cv2.resize(im, (WIDTH, HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # copy to draw on
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)

        # predict on the image
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(image, axis=0))

        # Variable initialization
        compteur = 0
        current_grp = 0

        # Stores the anchor value for each line
        # anchor = [anchor_line1, anchor_line1, anchor_line1]
        anchor = [0, 0, 0]

        # Sorting the boxes by the y value of the top left coordinate
        infos = zip(boxes[0], scores[0], labels[0])
        infos_sorted = sorted(infos, key=lambda x: x[0][1])

        # infos_lines = [infos_line1, infos_line2, infos_line3]
        infos_lines = [[], [], []]

        # Passing through all the boxes
        for box, score, label in infos_sorted:

            y = box[1]

            # Filtering some boxes with a too low score and the -1 values that come from padding.
            score_threshold = 0.2
            if label != -1 and score > score_threshold:

                # To initate anchor
                if compteur == 0:
                    anchor[current_grp] = y

                # Goes into that condition when another group starts
                if abs(y - anchor[current_grp]) > 50:
                    current_grp += 1
                    if current_grp > 2:
                        print("The caracters have been grouped in more than 3 lines")
                        break
                    anchor[current_grp] = y

                 # Goes into that condition when still in same group
                if abs(y - anchor[current_grp]) < 50:
                    infos_lines[current_grp].append([box, score, label])
                    anchor[current_grp] = y

                # Drawing the detect boxes on the image
                color = label_color(label)
                b = box.astype(int)
                draw_box(draw, b, color=color)
                caption = "{} {:.3f}".format(
                    self.labels_to_names[label], score)
                draw_caption(draw, b, self.labels_to_names[label],)

                compteur += 1

        # Sorting each lines on the x coordinates.
        # (aren't we reading from left to right ?)
        infos_lines[0] = sorted(infos_lines[0], key=lambda x: x[0][0])
        infos_lines[1] = sorted(infos_lines[1], key=lambda x: x[0][0])
        infos_lines[2] = sorted(infos_lines[2], key=lambda x: x[0][0])

        # Printing text
        lines = ['', '', '']
        # Line 1
        for infos_carac in infos_lines[0]:
            lines[0] += self.labels_to_names[infos_carac[2]]
        # Line 2
        for infos_carac in infos_lines[1]:
            lines[1] += self.labels_to_names[infos_carac[2]]
        # Line 3
        for infos_carac in infos_lines[2]:
            lines[2] += self.labels_to_names[infos_carac[2]]

        return draw, lines


if __name__ == '__main__':
    """Just a sample test of chip detection"""

    im_path = '../dataset/img_test/'
    im_name = 'FULL-2019-04-26-160938.png'

    # Load the image in grayscale mode
    image = cv2.imread(im_path + im_name, 0)

    # Measuring inference time
    detector = CaracDetector()
    start = time.time()
    box_coord, score = detector.detect_chip(image)
    end = time.time()
    print('Inference time = ', end - start)

    print('Predicted angle =', score, box_coord)
