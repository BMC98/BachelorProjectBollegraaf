import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import os  
import math 

#This code can be applied after splitting the resulting array
#from kmeans_array.py into individual images.

path = "Some/Path/"
save_path = "Clustering_Masks"

if not os.path.exists(save_path):
    os.mkdir(save_path) 


images = [img for img in os.listdir(new_path) if img.endswith(".png")]

	
for img_name in images:
	image_path = os.path.join(new_path, img_name)
	img = cv.imread(image_path)
	

	maxfill = 1000
	img_bw = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(img_bw, 127, 255, 0)


	labels = labels.reshape(img.shape[0], img.shape[1]).astype(np.uint8)
	print("Thresh: ", thresh.shape, thresh.dtype)
	print("Labels: ", labels.shape, np.unique(labels))

	contours, hier = cv.findContours(labels, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
	if len(contours) > 0:
		small = [cv.contourArea(contours[p]) < maxfill for p in range(len(contours))]  
		contours = np.array(contours)[small]
			
	for cnt in contours:
		x, y, w, h = cv.boundingRect(cnt)

		ROI = labels[y: y + h, x : x + w]

		occurrences = np.bincount(ROI.flatten())
		occurences_no_background = occurrences[1:]
		most_occurring = np.argmax(occurences_no_background) + 1

		print(most_occurring)

		cv.drawContours(labels, [cnt], 0, int(most_occurring), -1)  # fill
			

	# for cnt in contours:  # only contains contours of holes
		# cv.drawContours(labels, [cnt], 0, 255, -1)  # fill

	# cv.imshow("labels", labels.astype(np.uint8))
	# cv.imshow("img", img)
	save_path_new = os.path.join(save_path, img_name)
	# cv.imwrite(save_path_new, img)
		
	# cv.waitKey(0)
	print(k)

	fig, (ax, ax2) = plt.subplots(1, 2)
	ax.imshow(img)
	ax2.imshow(labels)
	plt.show()

		
	# plt.show()
	# exit(0)
		