# coding: utf-8
import numpy as np
import cv2
import tkinter
from PIL import Image, ImageTk
from threading import Thread
import imutils

class GUI(Thread):

    def __init__(self, img):
        Thread.__init__(self)
        self.img = img
        self.root = tkinter.Tk()
        self.root.title("Result")
        #Rearrang the color channel
        self.frame = None
        #self.thread = None
        #self.stopEvent = None
        self.panel = None

        #w = tkinter.Label(self.root, text="Hello, world!")
        #w.pack()
        self.root.minsize(width=500,height=500)
        self.root.attributes("-fullscreen", True)
        

        #self.stopEvent = threading.Event()
        #self.thread = threading.Thread(target=self.displayImage, args=())

        # set a callback to handle when the window is closed
        self.root.wm_title("Result")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.callback)

        #self.root.after(1000,self.update_txt)
        #self.root.mainloop()
        self.start()
        self.root.mainloop()

    def run(self):
        while True:
            if self.img is not None:
                print("has to display image !")
                self.displayImage()

    
    def update_txt(self):
        print("update txt !")

    def callback(self):
        self.root.quit()

    def loadImage(self, filename, resize=None):
        image = Image.open(filename)
        if resize is not None:
            image = image.resize(resize, Image.ANTIALIAS)
        return ImageTk.PhotoImage(image)

    def displayImage(self):

        #self.img = imutils.resize(self.img, width=300)

        #tkimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        #tkimg = Image.fromarray(tkimg)
        #tkimg = ImageTk.PhotoImage(tkimg)
        tkimg = self.img

        # if the panel is not None, we need to initialize it
        if self.panel is None:
            self.panel = tkinter.Label(self.root, image=tkimg)
            self.panel.image = tkimg
            self.panel.pack(side="left", padx=10, pady=10)
        else:
            self.panel.configure(image=tkimg)
            self.panel.image = tkimg
        print("displayimage")

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        #self.stopEvent.set()
        self.root.quit()

result = GUI(None)
tkimage = result.loadImage("azureok.jpg")
result.img = tkimage
#result.displayImage(tkimage)