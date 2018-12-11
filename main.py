# coding: utf-8
#import camera
import detection_instance
import image
import gui
import cv2
import os
import imutils

def main():
    img = cv2.imread(os.path.join("./", '181206_115427_0000000005_CAM1_OK.bmp'))
    #cam = camera.Camera()
    detect = detection_instance.DetectionInstance(img)
    #image = Image(image)
    #gui = GUI(image)
    detect.get_chip_area()
    detect.get_text_orientations()
    detect.read_text()
    results = detect.text
    print (results)

    img = imutils.rotate(detect.chip, detect.orientation_used)

    #img = image.Image(img)
    #img = image.addSerialNumber(img, results)
    #image.saveImage(img)

    #cam.loadConf()
    #currImg, resizeImg = cam.grabbingImage()
    #cam.showImage(resizeImg)

    # detect.read_text()
    # résultat : detect.text
    # EAST_NET = cv2.dnn.readNet("../data/east.pb")
 
if __name__ == '__main__':
    main()
