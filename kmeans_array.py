import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import os  
import math 
from PIL import Image

def kmeans(img, save_path):

	##Import an image and convert it to color

	##Reshape the image to a 2D array and convert it to float
	pixel_values = img.reshape((-1,4))
	pixel_values = np.float32(pixel_values)
	#print(pixel_values.shape)

	"""
	Here we define the stopping criteria so that
	it stops either after 200 iterations or 
	when the clusters move by less than 0.1
	"""
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.3)

	k = 7

	_, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

	# convert back to 8 bit values
	centers = np.uint8(centers)

	# flatten the labels array
	labels = labels.flatten()

	# Here we convert all pixels to their respective centroid colours
	segmented_image = centers[labels.flatten()]

	#Reshape the image and save it
	segmented_image = segmented_image.reshape(img.shape)
	#save_path = os.path.join(save_path, img)
	print(save_path)
	print(np.unique(labels))
	#cv.imwrite(save_path, segmented_image)
	return labels
	#cv.waitKey(0)

path = "Fixed_Channels_Depth/"
save_path = "Clustering_Masks_Depth_Array7"

if not os.path.exists(save_path):
    os.mkdir(save_path) 



arr = np.load("train_images_nodepth.npy")
img = kmeans(arr, save_path)
img = np.reshape(img, (219, 512, 512))
for i in range(20):
	plt.imshow(img[i,:,:])
	plt.show()
np.save(save_path, img)


cv.waitKey(0)
	