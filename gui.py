# coding: utf-8
import numpy as np
import cv2
import tkinter
from PIL import Image, ImageTk
import threading
import imutils

class GUI(object):

	def __init__(self, img):
		self.img = img
		self.root = tkinter.Tk()
		self.root.title("Result")
		#Rearrang the color channel
		self.frame = None
		self.thread = None
		self.stopEvent = None
		self.panel = None

		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.displayImage, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("Result")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

	def loadImage(self, filename, resize=None):
	    image = Image.open(filename)
	    if resize is not None:
	        image = image.resize(resize, Image.ANTIALIAS)
	    return ImageTk.PhotoImage(image)

	def displayImage(self):
		try: 
			while not self.stopEvent.is_set():

				self.img = imutils.resize(self.img, width=300)

				tkimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
				tkimg = Image.fromarray(tkimg)
				tkimg = ImageTk.PhotoImage(tkimg)

				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tkinter.Label(image=tkimg)
					self.panel.image = tkimg
					self.panel.pack(side="left", padx=10, pady=10)
				else:
					self.panel.configure(image=tkimg)
					self.panel.image = tkimg
				self.root.mainloop()

		except RuntimeError as e:
			print("[INFO] caught a RuntimeError")

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.root.quit()

# result = GUI()
# tkimage = result.loadImage("../data/images/azureok.jpg", resize=(300,300))
# result.displayImage(tkimage)
