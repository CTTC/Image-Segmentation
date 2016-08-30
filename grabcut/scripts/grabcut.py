#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/chentao/Pictures/spatula.png')

drawing = False # true if mouse is pressed
rect_done = False
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix1,iy1 = -1,-1
ix2,iy2 = -1,-1

img_orig = img.copy()
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix1,iy1,ix2,iy2,drawing,mode, rect_done

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix1,iy1 = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            ix2,iy2 = x,y


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ix2,iy2 = x,y
        cv2.rectangle(img,(ix1,iy1),(ix2,iy2),(0,255,0),2)
        # ix1,iy1 = -1,-1   #uncoment this and comment the grabcut part when you do not use grabcut and want to draw multiple rectangles
        # ix2,iy2 = -1,-1
        rect_done = True



cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
	tempimg = img.copy()
	if drawing:
		if (ix1 != -1 and iy1 != -1 and ix2 != -1 and iy2 != -1):
		    cv2.rectangle(tempimg,(ix1,iy1),(ix2,iy2),(0,255,0),2)

	cv2.imshow('image',tempimg)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == 27:
		break

	if rect_done:
		print "done"
		break

mask = np.zeros(img.shape[:2],np.uint8)
 
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

ix1 = max(ix1,1)
iy1 = max(iy1,1)
ix2 = max(ix2,1)
iy2 = max(iy2,1)
ix1 = min(ix1,img_orig.shape[1])
iy1 = min(iy1,img_orig.shape[0])
ix2 = min(ix2,img_orig.shape[1])
iy2 = min(iy2,img_orig.shape[0])
# print "ix1: ",ix1
# print "iy1: ",iy1
# print "ix2: ",ix2
# print "iy2: ",iy2
rect = (ix1,iy1,ix2,iy2)
# rect = (13,11,202,194)
cv2.grabCut(img_orig,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
 
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img_orig = img_orig*mask2[:,:,np.newaxis]

while(1):
	cv2.imshow("Image",img_orig)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == 27:
		break