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
import yolo_char


import numpy as np
import tkinter
from PIL import Image, ImageTk
from threading import Thread
from PIL import Image

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
        #init detectors
        self.past_detection = yolo.YOLO()
        self.text_detection = yolo_text.YOLO()
        self.char_detection = yolo_char.YOLO()

        while True:

            #############################################PART 1##########################################
            #Grab image from camera
            fullimg, img = self.cam.grabbingImage()

            #img2 = img.copy()
            #start = time.time()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = cv2.imread(os.path.join("./", 'prothese.png'))

            #convert image to PIL format
            img_pil = Image.fromarray(fullimg)

            #is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_pil)
            #Init
            computeResults = image.Image(fullimg)
            #Pastille detection
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)

            #############################################PART 2##########################################
            #Text Detection
            if is_detected == True :

                detect = detection_instance.DetectionInstance(fullimg)
                is_cropped, img_chip = detect.get_chip_area(out_boxes)
                computeResults.saveImage(img_chip)
                print ("Image saved. Change/Turn prothesis")
                logging.info("Circle detected")
                print ("Image saved")
            else :
                print ("Unable to find circle. Please move the prosthesis")
                logging.info("Circle not found")

            self.ui.displayImage(img)

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

        #print("displayimage")

if __name__ == "__main__":
    app = ImplantBox()
