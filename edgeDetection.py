import cv2 as cv 
import numpy as np 
import os

read_path = 'Images'
save_path = 'Edges'

i=1

for item in os.listdir(read_path):
	print(item)
	filename = os.path.join(read_path, item)
	img = cv.imread(filename, cv.COLOR_BGR2GRAY)
	edge = cv.Canny(img, 100, 200)
	cv.imshow("Original", img)
	cv.imshow("Edges", edge)
	save_filename = 'Edges ' + str(i) + '.png'
	cv.imwrite(save_filename, edge)
	i+=1
	cv.waitKey(0)




