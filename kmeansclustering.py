#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/chentao/Pictures/bowl.png')

Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_PP_CENTERS

num_clusters = 2

# Apply KMeans
compactness, labels, centers = cv2.kmeans(Z, num_clusters, None, criteria = criteria, attempts = 50, flags = flags)

# print "compactness: "
# print compactness
# print "labels: "
# print labels
# print "centers: "
# print centers

# generating bright palette
colors = np.zeros((1, num_clusters, 3), np.uint8)
colors[0, :] = 255
colors[0, :, 0] = np.arange(0, 180, 180.0 / num_clusters)
colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0]

# Now convert back into uint8, and make original image
centers = np.uint8(centers)


clustered_img = colors[labels.flatten()]
clustered_img = clustered_img.reshape((img.shape))

cv2.imshow("Clustered Image", clustered_img)
cv2.waitKey(0)


# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img = cv2.imread('/home/chentao/Pictures/bowl.png')


# clustered_img = cv2.pyrMeanShiftFiltering(img, sp = 80, sr = 40)


# cv2.imshow("Clustered Image", clustered_img)
# cv2.waitKey(0)