#!/usr/bin/env python


#===========================================================
# The following codes can only extract the moving objects,
# Static objects are assumed to be background
#===========================================================

## ****************************************************************
#BackgroundSubtractorMOG
import numpy as np
import cv2
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow("Original", frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break
 
cap.release()
cv2.destroyAllWindows()


## ****************************************************************
# #BackgroundSubtractorGMG
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt


# cap = cv2.VideoCapture(0)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

#     cv2.imshow("Original", frame)

#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#        break
 
# cap.release()
# cv2.destroyAllWindows()