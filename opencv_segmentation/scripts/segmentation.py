#!/usr/bin/env python

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from distutils.version import LooseVersion

print('**Current Working Directory: {}'.format(os.getcwd()))
print('**OpenCV version: {}'.format(LooseVersion(cv2.__version__).version[0]))

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
cv2.namedWindow('Thresholded', cv2.WINDOW_NORMAL)
cv2.namedWindow('SureFG', cv2.WINDOW_NORMAL)
cv2.namedWindow('SureBG', cv2.WINDOW_NORMAL)
cv2.namedWindow('Unknown', cv2.WINDOW_NORMAL)
cv2.namedWindow('Markers', cv2.WINDOW_NORMAL)
cv2.namedWindow('Final', cv2.WINDOW_NORMAL)

img = cv2.imread('/home/chentao/Pictures/coins.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("Original",img)
cv2.imshow("Gray",gray)
cv2.imshow("Thresholded",thresh)

 # noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0) 
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow("SureBG", sure_bg)
cv2.imshow("SureFG", sure_fg)
cv2.imshow("Unknown", unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1


# Now, mark the region of unknown with zero
# for i in range(markers.shape[0]):
# 	for j in range(markers.shape[1]):
# 		print "  {}  ".format(markers[i][j]),
# 	print "\n"
markers[unknown==255] = 0
cv2.imshow("Markers", markers)

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
cv2.imshow("Final", img)
cv2.waitKey(20000)
cv2.destroyAllWindows()

