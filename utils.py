"""
Tools functions for text and circle detections
"""
import time
import urllib.parse
import urllib
import requests
import cv2
import numpy as np
import http.client
import logging

import imutils
from imutils.object_detection import non_max_suppression
import math
import keras.backend as K



def redress_boundingbox(offsets, x_data, angle):
    """
    Compute the coordinates of the bounding boxe around text
    from geometric data
    """
    # extract the rotation angle for the prediction and then
    # compute the sin and cosine
    cos = np.cos(angle)
    sin = np.sin(angle)

    # use the geometry volume to derive the width and height of
    # the bounding box
    height = x_data[0] + x_data[2]
    width = x_data[1] + x_data[3]

    # compute both the starting and ending (x, y)-coordinates for
    # the text prediction bounding box
    # with offset factor as our resulting feature maps will be 4x smaller than the input image
    end_x = int(offsets[0] * 4.0 + (cos * x_data[1]) + (sin * x_data[2]))
    end_y = int(offsets[1] * 4.0 - (sin * x_data[1]) + (cos * x_data[2]))
    start_x = int(end_x - width)
    start_y = int(end_y - height)

    # add the bounding box coordinates and probability score to
    # our respective lists
    return (start_x, start_y, end_x, end_y)

def detect_text(image, network, degree):
    """ Detect text position with EAST network
    """
    image = imutils.rotate(image, degree)

    old_shape = image.shape

    min_confidence = 0.5
    max_text_angle = 0.15

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (224, 224))
    width_ratio = old_shape[1] / float(image.shape[1])
    height_ratio = old_shape[0] / float(image.shape[0])

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    network.setInput(cv2.dnn.blobFromImage(
        image, 1.0, (image.shape[1], image.shape[0]), (123.68, 116.78, 103.94),
        swapRB=True, crop=False
    ))

    # the first layer is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    (scores, geometry) = network.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    rects = []
    confidences = []

    # loop over the number of rows
    for row_index in range(0, scores.shape[2]):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        # loop over the number of columns
        for col_index in range(0, scores.shape[3]):
            if scores[0, 0, row_index][col_index] < min_confidence:
                continue
            if abs(geometry[0, 4, row_index][col_index]) > max_text_angle:
                continue
            rects.append(redress_boundingbox(
                [col_index, row_index],
                [
                    geometry[0, 0, row_index][col_index], geometry[0, 1, row_index][col_index],
                    geometry[0, 2, row_index][col_index], geometry[0, 3, row_index][col_index]
                ],
                geometry[0, 4, row_index][col_index]
            ))
            confidences.append(scores[0, 0, row_index][col_index])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    if len(boxes) >= 1:
        boxes = boxes*(width_ratio, height_ratio, width_ratio, height_ratio)

    return boxes

def get_circles(image, param, minr=140, maxr=0):
    """ Get the circles coordinates of an image
    """

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    circ = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, param, 80, minRadius=minr, maxRadius=maxr)
    if circ is not None:
        circ = np.uint16(np.around(circ))
    return circ

def microsoft_detection_text(image_data):
    """ Send a request to microsoft text detection to read the serial number
    """
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': '23062af450144fe6b5c5b53c48c8d98e',
    }

    params = urllib.parse.urlencode({
        # Request parameters
        'mode': 'Printed',
    })
    try:
        conn = http.client.HTTPSConnection('westeurope.api.cognitive.microsoft.com')
        conn.request("POST", "/vision/v2.0/recognizeText?%s" % params, image_data, headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    time.sleep(1)
    i = 0
    while True:
        if i > 20:
            break
        try:
            #conn = http.client.HTTPSConnection('westeurope.api.cognitive.microsoft.com')
            #conn.request("GET", response.getheader("Operation-Location")[46:], url_json, headers)
            res = requests.get(response.getheader("Operation-Location"), headers=headers)
            #response2 = conn.getresponse()
            data = res.json()
            if data['status'] == 'Succeeded':
                break
            time.sleep(0.2)
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
    return data
