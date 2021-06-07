import matplotlib.pyplot as plt 
import cv2 as cv
import os
from PIL import Image 

path = "Fixed_Channels"
save_path = "Fixed_Channels_Cropped1"

if not os.path.exists(save_path):
	os.mkdir(save_path)

images = [img for img in os.listdir(path) if img.endswith(".png")]

for img_name in images:
	image_path = os.path.join(path, img_name)
	img = cv.imread(image_path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	file_name = os.path.join(save_path, img_name)
	img = img[0:512, 73:450]
	cv.imwrite(file_name, img)
	cv.waitKey(0)