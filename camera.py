# coding: utf-8
import pypylon.pylon as py
import numpy as np
import cv2
import time
import uuid

class Camera(object):

    def __init__(self):
        self.last_timestamp = 0
        self.timestamp = 0
        self.conf = "NodeMap.pfs"
        self.first_device = py.TlFactory.GetInstance().CreateFirstDevice()
        self.instant_camera = py.InstantCamera(self.first_device)
        self.instant_camera.Open()

    def loadConf(self):
        py.FeaturePersistence.Load(self.conf, self.instant_camera.GetNodeMap())
        return "Config Loaded"

    def saveConf(self):
        py.FeaturePersistence.Save(self.conf, self.instant_camera.GetNodeMap())
        return "Config Saved"

    def grabbingImage(self):
        self.instant_camera.StartGrabbing(py.GrabStrategy_LatestImages)
        grabResult = self.instant_camera.RetrieveResult(5000, py.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            currImg = grabResult.Array
            resizeImg = cv2.resize(currImg, (1920, 1080))
            #resizeImg = resizeImg.astype(np.uint8)


        grabResult.Release()
        #cv2.imshow('Video', resizeImg)
        #cv2.waitKey(1000)
        return currImg, resizeImg

    def saveImage(self, resizeImg):
        cv2.imwrite( "./" + str(uuid.uuid4().hex) + ".jpg", resizeImg )
        return "Image Saved"

    def generateUUID():
        generated_uuid = uuid.uuid4().hex
        return generated_uuid

    def showImage(self):
        cv2.imshow('Video', resizeImg)
        cv2.waitKey(10000)

# image = Camera()
# img1, img2 = image.grabbingImage()
# print (len(img1))
# print (len(img2))
# image.saveImage(img2)

