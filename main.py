# coding: utf-8
#import camera
import detection_instance
import image
import gui
import cv2
import os
import imutils
import time
import logging

#LOGGER = logging.getLogger(__name__)

def main():

    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(filename='./activity.log', format=FORMAT, level=logging.DEBUG)

    while True:
        start = time.time()
        img = cv2.imread(os.path.join("./", '181206_115427_0000000005_CAM1_OK.bmp'))
        #img = cv2.imread(os.path.join("./", 'CAM1_6.bmp'))
        #cam = camera.Camera()
        detect = detection_instance.DetectionInstance(img)
        print('init: %0.3f'% (time.time()-start))
        #image = Image(image)
        #gui = GUI(image)
        start = time.time()
        if detect.get_chip_area() :
            logging.info("Circle detected")
            detect.get_text_orientations()
            logging.info("Picture Redressed")
            detect.read_text()
            print('process: %0.3f'% (time.time()-start))
            results = detect.text
            logging.info("Text read : %s"% results)
            print (results)

            img = imutils.rotate(detect.chip, detect.orientation_used)
            print (img.shape)

            computeResults = image.Image(img)
            img = computeResults.addSerialNumber(results)
            logging.info("Serial Number written")
            computeResults.saveImage()
            logging.info("Image saved")
        else :
            logging.info("Circle not found")

    #dispResults = gui.GUI(img)
    #img = dispResults.convertToTkImage()
    #dispResults.displayImage()
    #dispResults = image.addSerialNumber(img, results)
    #image.saveImage(img)

    #cam.loadConf()
    #currImg, resizeImg = cam.grabbingImage()
    #cam.showImage(resizeImg)

if __name__ == '__main__':
    main()
