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
        self.char_detection = yolo_char.YOLO()

        while True:

            #Grab image from camera
            fullimg, img = self.cam.grabbingImage()

            #img2 = img.copy()
            start = time.time()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = cv2.imread(os.path.join("./", 'val4.png'))

            #convert image to PIL format
            img_pil = Image.fromarray(img)
            
            #is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_pil)
            #Init
            computeResults = image.Image(img)
            #Pastille detection
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)
            print (out_boxes)

            #computeResults = image.Image(img_pil)

            #image1 = img[int(out_boxes[0][0]):int(out_boxes[0][2]), int(out_boxes[0][1]):int(out_boxes[0][3])]
            #image2 = img[int(out_boxes[1][0]):int(out_boxes[1][2]), int(out_boxes[1][1]):int(out_boxes[1][3])]
            #image3 = img[int(out_boxes[2][0]):int(out_boxes[2][2]), int(out_boxes[2][1]):int(out_boxes[2][3])]

            #computeResults.saveImage(img_pil, image1)
            #computeResults.saveImage(img_pastille, img_pastille)

            #is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(img_pil)
            #print (out_boxes)

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

                detect = detection_instance.DetectionInstance(img)
                is_cropped, img_chip = detect.get_chip_area()
                #image_pastille = img[int(out_boxes[0][0]):int(out_boxes[0][2]), int(out_boxes[0][1]):int(out_boxes[0][3])]
                computeResults.saveImage(img_chip)
                #print ("Pastille detectée, image enregistrées. Changez/Tournez la prothèse")
                time.sleep(3)
                logging.info("Circle detected")

                img_chip_pil = Image.fromarray(img_chip)

                is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(img_chip_pil)

                                #texte1, texte2, texte3 = detect.get_text_area(pil_img_pastille, out_boxes)

                for deg in range(0, 360, 5):
                    is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(pil_img_pastille)
                    #nb_max_text_detection = len(out_boxes)
                    if len(out_boxes) == 3:
                        img_text1, img_text2, img_text3 = detect.get_text_area(out_boxes)
                        all_text_is_detected = True
                        computeResults.saveImage(img_text1)
                        computeResults.saveImage(img_text2)
                        computeResults.saveImage(img_text3)
                    else :
                        pil_img_pastille.rotate(deg)
                        print ("Image Rotate...")
                        print (deg)
                        all_text_is_detected = False

                if all_text_is_detected == True :
                    img_text1_pil = Image.fromarray(img_text1)
                    img_text2_pil = Image.fromarray(img_text2)
                    img_text3_pil = Image.fromarray(img_text3)

                    is_detected_t1, out_boxes_t1, out_scores_t1, out_classes_t1 = self.char_detection.detect_image(img_text1_pil)
                    is_detected_t2, out_boxes_t2 out_scores_t2, out_classes_t2 = self.char_detection.detect_image(img_text2_pil)
                    is_detected_t3 out_boxes_t3 out_scores_t3, out_classes_t3 = self.char_detection.detect_image(img_text3_pil)

                    lines[0] = str(out_classes_t1)
                    lines[1] = str(out_classes_t2)
                    lines[2] = str(out_classes_t3)

                    print (lines[0])
                    print (lines[1])
                    print (lines[2])

                    if (lines != None) :
                        for line in lines :
                            for char in line:
                                self.keyboard.send(r)
                                self.keyboard.send("  ")

                            final_img = computeResults.addSerialNumber(lines)
                            logging.info("Serial Number written")
                            #cam.saveImage(fullimg, img)
                            computeResults.saveImage(final_img)
                            logging.info("Image saved")
                    else :
                        continue

                else :
                    print ("Unable to find all text area. Please move the prosthesis")
                    logging.info("All texts not found")



                #detect = detection_instance.DetectionInstance(img, out_boxes)
                #opencvImage = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
                #is_cropped, img_pastille = detect.get_chip_area()
                #print (img_pastille)
                #logging.info("Picture cropped")

                #computeResults = image.Image(img_pastille)
                #computeResults.saveImage(img2, img_pastille)

                #print ("detection de texte :")

                #pil_img_pastille = Image.fromarray(img_pastille)
                # computeResults = image.Image(img_pastille)
                # computeResults.saveImage(img2, img_pastille)

                #is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(pil_img_pastille)
                #is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(pil_img_pastille)
                #print (out_boxes)
                #computeResults.saveImage(image1, image2)
                #computeResults.saveImage(image3, img_pastille)


                #texte1, texte2, texte3 = detect.get_text_area(pil_img_pastille, out_boxes)

                # for deg in range(0, 360, 5):
                #     is_detected, out_boxes, out_scores, out_classes = self.text_detection.detect_image(pil_img_pastille)
                #     if len(out_boxes) >= 1:
                #         print ("text detection ok")
                #         print (out_boxes)
                #         #texte1, texte2, texte3 = detect.get_text_area(pil_img_pastille, out_boxes)
                #         computeResults.saveImage(text1, texte2)
                #     else :
                #         pil_img_pastille.rotate(deg)
                #         print ("Image Rotate...")
                #         print (deg)



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
                print ("Unable to find circle. Please move the prosthesis")
                logging.info("Circle not found")
                #time.sleep(1)
            
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
