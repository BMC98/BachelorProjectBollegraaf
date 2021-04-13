import cv2 as cv
import matplotlib.pyplot as plt

def rescaleFrame(frame,scale):
	width = int(frame.shape[1] * scale)
	height = int(frame.shape[0] * scale)

	dimensions = (width,height)

	return cv.resize(frame,dimensions, interpolation = cv.INTER_AREA) 

img = cv.imread('Photos/1.png')
img_resized = rescaleFrame(img,0.75)

cv.imshow('Figure',img)
cv.imshow('Rescaled Figure',img_resized)

cv.waitKey(0)