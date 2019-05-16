from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import time
import imutils
from utils import angle_error
import cv2
import matplotlib.pyplot as plt



def classify_angle(im, model):

	"""Classifies an image (np array or keras array)
	See here for difference:
	https://stackoverflow.com/questions/53718409/numpy-array-vs-img-to-array

	Parameters:
	im: nummpy array
	The image to classify

	Returns:
	angle : int
	in [0, 1, 2, : 360] corresponding to the angle in a counter clockwise direction.
	
	im_corr : np array
	The image with the orientation corrected


	"""

	im_b = im/255
	im_b = np.expand_dims(im_b, axis=0)# correct shape for classification

	classe = model.predict(im_b)

	classe =classe.argmax(axis = 1) #taking index of the maximum %

	angle = classe[0]

	im_corr = imutils.rotate(im, -1* angle)


	return angle, im_corr






def main():
	HEIGHT = 224
	WIDTH = 224

	model = load_model('../models/rotnet_chip_resnet50.hdf5',custom_objects={'angle_error': angle_error})
	im_path = '../ds/ds_rotated/test_vrac/'
	im = 'FULL-2019-04-26-140952.png'

	image = cv2.imread(im_path+im)


	plt.imshow(image)
	plt.show()


	start = time.time() # Measuring inference time

	predicted_orientation, im_corr = classify_angle(image,model)

	end = time.time()
	print('Inference time = ',end - start)

	print('Predicted angle =', predicted_orientation)
	plt.imshow(im_corr)
	plt.show()
