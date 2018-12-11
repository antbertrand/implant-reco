import time
import requests
import http
import urllib
import imutils
import cv2
import numpy as np
import http.client, urllib.request, urllib.parse, urllib.error, base64, json

from imutils.object_detection import non_max_suppression

EAST_NET = cv2.dnn.readNet("../data/east.pb")

class DetectionInstance(object):
    def __init__(self, frame):
        self.frame = frame
        self.chip_crop_size = (200, 200)
        self.chip = None
        self.text_orientations = []
        self.orientation_used = None
        self.text = None
        self.network = EAST_NET

    def _detect_text(self, image, degree):
        crop_img = imutils.rotate(image, degree)

        (oldH, oldW) = crop_img.shape[:2]

        min_confidence = 0.5

        # set the new width and height and then determine the ratio in change
        # for both the width and height

        # resize the image and grab the new image dimensions
        image = cv2.resize(crop_img, (192, 192))
        (H, W) = image.shape[:2]
        rW = oldW / float(W)
        rH = oldH / float(H)

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        net = self.network

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        # show timing information on text prediction

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        if len(boxes)>0:
            boxes = boxes*(rW, rH, rW, rH)

        return boxes

    def _get_circles(self, image):
        """ Get the circles coordinates of an image
        """
        if len(image.shape)>2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circ = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, 80, minRadius=40, maxRadius=500)
        if circ is not None:
            circ = np.uint16(np.around(circ))
        return circ

    def get_chip_area(self):
        """ Search for a circle to get the coordinates of the chip
        """
        img = imutils.resize(self.frame, height=600)
        xratio = self.frame.shape[1]/img.shape[1]
        yratio = self.frame.shape[0]/img.shape[0]

        circles = self._get_circles(img)

        for circle in circles[0,:]:
            #if circle[2]>120:
            #    continue
            cv2.circle(img, (circle[0], circle[1]), circle[2], (0,255,0), 2)
            cv2.circle(img, (circle[0], circle[1]), 2, (0,0,255), 3)
            real_circle = circle.copy()

        cropCoords = (max(0, real_circle[1]-self.chip_crop_size[0]//2)*yratio,min(img.shape[0], real_circle[1]+self.chip_crop_size[0]//2)*yratio,
                      max(0, real_circle[0]-self.chip_crop_size[1]//2)*xratio,min(img.shape[1], real_circle[0]+self.chip_crop_size[1]//2)*xratio)

        image = self.frame[int(cropCoords[0]):int(cropCoords[1]), int(cropCoords[2]):int(cropCoords[3])]

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)

        self.chip = image

        return True

    def get_text_orientations(self):
        """ Search for 3 text areas in the chip area
        """
        image = cv2.cvtColor(self.chip, cv2.COLOR_GRAY2BGR)
        for deg in range(0, 360, 20):
            boxes = self._detect_text(image, deg)
            if len(boxes)>2:
                self.text_orientations.append(deg)

    def _microsoft_detection_text(self, image_data):
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

    def read_text(self):
        start = time.time()

        text = []
        for degree in self.text_orientations:
            image = imutils.rotate(self.chip, degree)

            image_data = cv2.imencode('.jpg', image)[1].tostring()
            data = self._microsoft_detection_text(image_data)

            flag_AA = False
            if not 'recognitionResult' in data:
                print(data)
            else:
                for line in data['recognitionResult']['lines']:
                    if line['text'].startswith('AA'):
                        flag_AA = True
                        text = []
                    if flag_AA:
                        text.append(line['text'])
                if len(text) == 3:
                    self.text = text
                    self.orientation_used = degree
                    break
