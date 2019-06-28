import os
import cv2
import glob



for name in sorted(glob.glob('/Users/cdidriche/Downloads/Eurosilicone_Pastilles_20Mpx/ds/img/*')):
	print(name)

	img = cv2.imread(name)
	print (img)
	if img is None :
		#print (img)
		continue
	else :
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		equ = cv2.equalizeHist(img)
		cv2.imwrite(name, equ)
