import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

##Import an image and convert it to color

img = cv.imread('Images/4.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


##Reshape the image to a 2D array and convert it to float
pixel_values = img.reshape((-1,3))
pixel_values = np.float32(pixel_values)
#print(pixel_values.shape)

"""
Here we define the stopping criteria so that
it stops either after 200 iterations or 
when the clusters move by less than 0.1 (TO BE CHANGED)
"""
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.3)

#5 clusters, for the 4 classes + the background (NEEDS REVISION)
k = 5
_, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# Here we convert all pixels to their respective centroid colours
segmented_image = centers[labels.flatten()]


segmented_image = segmented_image.reshape(img.shape)
plt.imshow(segmented_image)
plt.show()

cv.waitKey(0)