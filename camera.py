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
        self.first_device = py.TlFactory.GetInstance().CreateFirstDevice()
        self.instant_camera = py.InstantCamera(self.first_device)
        self.instant_camera.Open()
        #self.instant_camera.PixelFormat = 'RGB8'

    def loadConf(self, confname="NodeMap.pfs"):
        py.FeaturePersistence.Load(confname, self.instant_camera.GetNodeMap())
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
        self.instant_camera.StopGrabbing()
        #cv2.imshow('Video', resizeImg)
        #cv2.waitKey(1000)
        return currImg, resizeImg

    def saveImage(self, currImg, resizeImg):
        uuid = Camera.generateUUID()
        #Image.fromarray(resizeImg).save("./" + str(uuid.uuid4().hex) + ".tiff")
        cv2.imwrite( "./" + uuid + "resized" + ".png", cv2.cvtColor(resizeImg, cv2.COLOR_RGB2BGR) )
        cv2.imwrite( "./" + uuid + "full" + ".png", cv2.cvtColor(currImg, cv2.COLOR_RGB2BGR) )
        return "Images Saved"

    def generateUUID():
        generated_uuid = str(uuid.uuid4().hex)
        return generated_uuid

    def showImage(self):
        cv2.imshow('Video', resizeImg)
        cv2.waitKey(10000)

#image = Camera()
#img1, img2 = image.grabbingImage()
#print (len(img1))
#print (len(img2))
#image.saveImage(img1, img2)

