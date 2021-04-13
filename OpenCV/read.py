import cv2 as cv

img = cv.imread('Photos/1.png')

"""
plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.show()
"""

cv.imshow('Figure 1',img)

cv.waitKey(0)