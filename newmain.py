# coding: utf-8
import camera
import detection_instance
import image
import cv2
import os
import imutils
import time
import logging
from keyboard import Keyboard
import yolo
import yolo_text


import numpy as np
import tkinter
from PIL import Image, ImageTk
from threading import Thread

########################################################################
class Main(Thread):

    def __init__(self, ui):
        Thread.__init__(self)

        self.ui = ui

        FORMAT = '%(asctime)s %(message)s'
        logging.basicConfig(filename='./activity.log', format=FORMAT, level=logging.DEBUG)
        self.cam = camera.Camera()
        #self.past_detection = yolo.YOLO()
        logging.info("Camera detected")
        #self.cam.saveConf()
        #logging.info("Camera configuration saved")
        self.cam.loadConf()
        logging.info("Camera configuration loaded")
        self.keyboard = Keyboard()
        self.keyboard.openAndListen()
 
    #----------------------------------------------------------------------
    def run(self):
        self.past_detection = yolo.YOLO()
        self.text_detection = yolo_text.YOLO()

        while True:

            fullimg, img = self.cam.grabbingImage()
            img2 = img.copy()
            start = time.time()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)
            print (out_boxes)
            #print (out_boxes[0][0])
            #print (type(img2))
            #print (img2)
            #print(img.shape)
            #img2 = cv2.imread(os.path.join("./", '181213_102435_0000000008_CAM1_OK.bmp'))
            #img = cv2.imread(os.path.join("./", 'CAM1_6.bmp'))

            #detect = detection_instance.DetectionInstance(img2)
            #detect = detection_instance.DetectionInstance(img)

            print('init: %0.3f'% (time.time()-start))
            #image = Image(image)
            #gui = GUI(image)
            start = time.time()
            if is_detected == True :
                logging.info("Circle detected")
                detect = detection_instance.DetectionInstance(img2, out_boxes)
                #opencvImage = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
                is_cropped, img_pastille = detect.get_chip_area()
                #print (img_pastille)
                logging.info("Picture cropped")

                computeResults = image.Image(img_pastille)
                computeResults.saveImage(img2, img_pastille)

                is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_pil)


                print ("CHANGER PROTHESE")
                time.sleep(5)

                # # detect.get_text_orientations()
                # # logging.info("Picture Redressed")
                # # detect.read_text()
                # print('process: %0.3f'% (time.time()-start))
                # results = detect.text
                # logging.info("Text read : %s"% results)
                # print (results)
                # logging.info("Keystrokes : %s"% results)
                # if (results != None) :
                #     for r in results:
                #         self.keyboard.send(r)
                #         self.keyboard.send("	")

                #     img = imutils.rotate(detect.chip, detect.orientation_used)
                #     #print (img.shape)

                #     computeResults = image.Image(img)
                #     img = computeResults.addSerialNumber(results)
                #     logging.info("Serial Number written")
                #     #cam.saveImage(fullimg, img)
                #     computeResults.saveImage(fullimg, img)
                #     logging.info("Image saved")
                # else :
                #     continue
            else :
                logging.info("Circle not found")
                time.sleep(1)
            
            #self.ui.img = self.ui.loadImage(img)
            self.ui.displayImage(img)
            #dispResults = gui.GUI(img)
            #img = dispResults.convertToTkImage(img)
            #dispResults.displayImage()




########################################################################
class ImplantBox():
 
    #----------------------------------------------------------------------
    def __init__(self):
        Thread.__init__(self)
        self.img = None
        self.root = tkinter.Tk()
        self.root.title("Result")
        # Rearrange the color channel
        self.frame = None
        self.panel = None

        self.root.minsize(width=500,height=500)
        self.root.attributes("-fullscreen", True)

        # set a callback to handle when the window is closed
        self.root.wm_title("Result")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.callback)

        main = Main(self)
        main.start()

        self.root.mainloop()

    def callback(self):
        self.root.quit()

    def loadImage(self, filename, resize=None):
        image = Image.open(filename)
        if resize is not None:
            image = image.resize(resize, Image.ANTIALIAS)
        return ImageTk.PhotoImage(image)

    def displayImage(self, img):
        img = imutils.resize(img, width=300)

        # tkimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # tkimg = Image.fromarray(tkimg)
        # tkimg = ImageTk.PhotoImage(tkimg)

        #tkimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tkimg = Image.fromarray(img)
        tkimg = ImageTk.PhotoImage(tkimg)

        # if the panel is not None, we need to initialize it
        if self.panel is None:
            self.panel = tkinter.Label(self.root, image=tkimg)
            self.panel.image = tkimg
            self.panel.pack(side="left", padx=10, pady=10)
        else:
            self.panel.configure(image=tkimg)
            self.panel.image = tkimg

        print("displayimage")

if __name__ == "__main__":
    app = ImplantBox()
