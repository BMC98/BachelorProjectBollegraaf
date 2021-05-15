import time
import os
import cv2
import numpy as np
import imageio


#labelmap_path = "../training_demo/annotations/simple_label_map.txt"
path = "Images/"
savepath = "Fixed_Channels"

if not os.path.exists(savepath):
    os.mkdir(savepath) 

images = [img for img in os.listdir(path) if img.endswith(".png")]

for img_name in images:
    # image_np = cv2.imread(path + img_name).astype(np.uint8)
    image_raw = imageio.imread(path + img_name).astype(np.uint8)
    image_np = np.zeros((image_raw.shape[0], int(image_raw.shape[1] / 4), 3), dtype = np.uint8)
    image_np[:,:,2] = image_raw[:,0:512]
    image_np[:,:,1] = image_raw[:,512:1024]
    image_np[:,:,0] = image_raw[:,1024:1536]

    filename = os.path.join(savepath, img_name)
    print(filename)
    cv2.imwrite(filename, image_np) 