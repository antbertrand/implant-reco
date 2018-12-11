# coding: utf-8
import tkinter as tk
from PIL import Image, ImageTk

class GUI:

	def __init__(self):
		self.fenetre = tk.Tk()
		self.fenetre.title("Result")

	def loadImage(self, filename, resize=None):
	    image = Image.open(filename)
	    if resize is not None:
	        image = image.resize(resize, Image.ANTIALIAS)
	    return ImageTk.PhotoImage(image)

	def displayImage(self, tkimage):
		label = tk.Label(self.fenetre, image=tkimage)
		label.pack()
		self.fenetre.mainloop()

# result = GUI()
# tkimage = result.loadImage("../data/images/azureok.jpg", resize=(300,300))
# result.displayImage(tkimage)