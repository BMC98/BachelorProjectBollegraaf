import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color, morphology, io
from skimage.segmentation import flood, flood_fill


img = io.imread('Images/2.png')


flooded = flood_fill(img, (292, 171), 255, tolerance = 10)
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].imshow(img, cmap = plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis = 'off'

ax[1].imshow(flooded, cmap = plt.cm.gray)
ax[1].set_title('Flooded')
ax[1].axis = 'off'

plt.show()