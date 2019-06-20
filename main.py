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
            ##fullimg, img = self.cam.grabbingImage()

            #img2 = img.copy()
            #start = time.time()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.imread(os.path.join("./", 'prothese.png'))

            #convert image to PIL format
            img_pil = Image.fromarray(img)
            
            #is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_pil)
            #Init
            computeResults = image.Image(img)
            #Pastille detection
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)
            #print (out_boxes)
            #image = Image(image)
            #gui = GUI(image)

            #############################################PART 2##########################################
            #Text Detection
            if is_detected == True :

                detect = detection_instance.DetectionInstance(img)
                is_cropped, img_chip = detect.get_chip_area(out_boxes)
                computeResults.saveImage(img_chip)
                #print ("Pastille detectée, image enregistrées. Changez/Tournez la prothèse")
                logging.info("Circle detected")

                #Opencv to PILLOW image
                img_chip_pil = Image.fromarray(img_chip)
                img_chip_pil_rotate = img_chip_pil.copy()

                #init array results
                best_scores = np.array([0,0,0], dtype=float)
                best_boxes = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
                best_deg = np.array([0,0,0], dtype=int)

                #check every image rotate, with 10 deg step
                for deg in range(0, 360, 10):
                    #inference function
                    is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_chip_pil_rotate)
                    #Check the best score detection for each text
                    if len(out_scores) == 3:
                        for i in range(len(out_scores)):
                            if out_scores[i] > best_scores[i]:
                                best_scores[i] = out_scores[i]
                                best_deg[i] = deg

                                for y in range(len(out_boxes[i])):
                                    best_boxes[i][y] = out_boxes[i][y]

                        #rotate image before saving to visualize... (For dev)
                        img_chip_pil_rotate = img_chip_pil.rotate(deg)
                        open_cv_image = np.array(img_chip_pil_rotate) 
                        computeResults.saveImage(open_cv_image)

                    else :
                        continue

                if len(best_scores) == 3:
                    #Crop texts detection
                    img_text1, img_text2, img_text3 = detect.get_text_area(best_boxes, best_deg[0], best_deg[1], best_deg[2])

                    #Save texte images
                    computeResults.saveImage(img_text1)
                    computeResults.saveImage(img_text2)
                    computeResults.saveImage(img_text3)

                    #Convert to PIL format
                    img_text1_pil = Image.fromarray(img_text1)
                    img_text2_pil = Image.fromarray(img_text2)
                    img_text3_pil = Image.fromarray(img_text3)

                    #############################################PART 3##########################################
                    #Char detection
                    #Init list of char
                    list_line1 = []
                    list_line2 = []
                    list_line3 = []

                    #inference function
                    is_detected_t1, out_boxes_t1, out_scores_t1, out_classes_t1 = self.char_detection.detect_image(img_text1_pil)
                    is_detected_t2, out_boxes_t2, out_scores_t2, out_classes_t2 = self.char_detection.detect_image(img_text2_pil)
                    is_detected_t3, out_boxes_t3, out_scores_t3, out_classes_t3 = self.char_detection.detect_image(img_text3_pil)

                    #if at least 1 detection by line...
                    if (is_detected_t1 == True and is_detected_t2 == True and is_detected_t3 == True):
                        for char in out_classes_t1:
                            list_line1.append(char)
                        for char in out_classes_t2:
                            list_line2.append(char)
                        for char in out_classes_t3:
                            list_line3.append(char)

                        #convert from list to concatenated string
                        list_line1_str = ''.join(map(str, list_line1))
                        list_line2_str = ''.join(map(str, list_line2))
                        list_line3_str = ''.join(map(str, list_line3))

                        final_list = [list_line1_str, list_line2_str, list_line3_str]

                        #Instructions for sending to Arduino and simulate keystrokes of a keyboard...
                        #Send string by string
                        self.keyboard.send(list_line1_str)
                        self.keyboard.send("  ")
                        self.keyboard.send(list_line2_str)
                        self.keyboard.send("  ")
                        self.keyboard.send(list_line3_str)

                        #add Serial Number on the final picture
                        final_img = computeResults.addSerialNumber(final_list)
                        logging.info("Serial Number written")
                        #cam.saveImage(fullimg, img)
                        computeResults.saveImage(final_img)
                        logging.info("Image saved")

                    else :
                        print ("Unable to find all char. Please move the prosthesis")
                        logging.info("All char not found")
                else :
                    print ("Unable to find all text area. Please move the prosthesis")
                    logging.info("All texts not found")
            else :
                print ("Unable to find circle. Please move the prosthesis")
                logging.info("Circle not found")
                #time.sleep(1)
            
            #self.ui.img = self.ui.loadImage(img)
            self.ui.displayImage(img)
            #print('init: %0.3f'% (time.time()-start))
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

        #print("displayimage")

if __name__ == "__main__":
    app = ImplantBox()
