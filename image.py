# coding: utf-8
import numpy as np
import cv2
import time
import uuid

class Image(object):

    def __init__(self, img):
        self.img = img
        self.path = "./"

    def openImage(self, filename):
        img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        return img

    def saveImage(self, currImg, resizeImg):
        uuid = Image.generateUUID()
        #Image.fromarray(resizeImg).save("./" + str(uuid.uuid4().hex) + ".tiff")
        cv2.imwrite( "./" + uuid + "resized" + ".png", cv2.cvtColor(resizeImg, cv2.COLOR_RGB2BGR) )
        cv2.imwrite( "./" + uuid + "full" + ".png", currImg)
        return "Images Saved"

    def generateUUID():
        generated_uuid = str(uuid.uuid4().hex)
        return generated_uuid

    def addSerialNumber(self, serialNumber):

        if len(self.img.shape) < 3 :
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale              = 2
        fontColor              = (0,255,0)
        lineType               = 3
        i = 0 
        for serial in serialNumber:
            cv2.putText(self.img,serial,
            (10,50+(i*60)),
            font,
            fontScale,
            fontColor,
            lineType)
            i+=1
        return self.img

# image = Image()
# img  = image.openImage("../data/images/azureok.jpg")
# serialnumber = ["AA178", "81/350cc", "M0904"]
# img = image.addSerialNumber(img, serialnumber)
# path = "./"
# image.saveImage(img, path)
