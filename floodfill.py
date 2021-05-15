import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color, morphology, io
from skimage.segmentation import flood, flood_fill
from skimage.color import rgb2gray

img = io.imread('Images/2.png')

img_sobel = filters.sobel(img)

img_flood = flood(img_sobel, (240,265), tolerance = 0.05)

floodfilled = flood_fill(img, (292, 171), 255, tolerance = 10)

fig, ax = plt.subplots(ncols=4, figsize=(40, 40))


ax[0].imshow(img, cmap = plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis = 'off'


ax[1].imshow(img_sobel)
ax[1].set_title('Sobel')
ax[1].axis = 'off'

ax[2].imshow(img_flood)
ax[2].set_title('Flooded')
ax[2].axis = 'off'

ax[3].imshow(floodfilled)
ax[3].set_title('Floodfilled')
ax[3].axis = 'off'

fig.tight_layout()
plt.show()