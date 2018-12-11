# coding: utf-8
import numpy as np
import cv2
import time
import uuid

class Image:

    def __init__(self):
        self.path = "./"

    def openImage(self, filename):
        img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        return img

    def saveImage(self, resizeImg, path="./"):
        cv2.imwrite(str(path) + str(uuid.uuid4().hex) + ".jpg",resizeImg)
        return "Image Saved"

    def generateUUID():
        generated_uuid = uuid.uuid4().hex
        return generated_uuid

    def addSerialNumber(self, image, serialNumber):

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale              = 2
        fontColor              = (0,255,0)
        lineType               = 3
        i = 0 
        for serial in serialNumber:
            cv2.putText(image,serial,
            (10,50+(i*60)),
            font,
            fontScale,
            fontColor,
            lineType)
            i+=1
        return image

# image = Image()
# img  = image.openImage("../data/images/azureok.jpg")
# serialnumber = ["AA178", "81/350cc", "M0904"]
# img = image.addSerialNumber(img, serialnumber)
# path = "./"
# image.saveImage(img, path)
