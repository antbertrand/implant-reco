# coding: utf-8
import camera
import detection_instance
import image
import gui
import cv2
import os
import imutils
import time
import logging
from keyboard import Keyboard

#LOGGER = logging.getLogger(__name__)

def main():

    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(filename='./activity.log', format=FORMAT, level=logging.DEBUG)
    cam = camera.Camera()
    logging.info("Camera detected")
    cam.saveConf()
    logging.info("Camera configuration saved")
    #cam.loadConf()
    #logging.info("Camera configuration loaded")
    k = Keyboard()
    k.openAndListen()

    while True:
        fullimg, img = cam.grabbingImage()
        start = time.time()
        #print(img.shape)
        #img2 = cv2.imread(os.path.join("./", '181213_102435_0000000008_CAM1_OK.bmp'))
        #img = cv2.imread(os.path.join("./", 'CAM1_6.bmp'))

        #detect = detection_instance.DetectionInstance(img2)
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
            logging.info("Keystrokes : %s"% results)
            for r in results:
            	k.send(r)
            	k.send("	")

            img = imutils.rotate(detect.chip, detect.orientation_used)
            print (img.shape)

            computeResults = image.Image(img)
            img = computeResults.addSerialNumber(results)
            logging.info("Serial Number written")
            #cam.saveImage(fullimg, img)
            computeResults.saveImage(fullimg, img)
            logging.info("Image saved")
        else :
            logging.info("Circle not found")
            time.sleep(1)
        #dispResults = gui.GUI(img)
        #img = dispResults.convertToTkImage(img)
        #dispResults.displayImage()
    #dispResults = image.addSerialNumber(img, results)
    #image.saveImage(img)

if __name__ == '__main__':
    main()
