import cv2 as cv
import numpy as np 

blank = np.zeros((900,900,3), dtype = 'uint8')


#1. Paint the image a certain colour
blank[:] = 0,0,0

#2. Draw a rectangle
cv.rectangle(blank, (0,0),(250,250), (0,255,0), thickness = 2)

#3. Draw a circle
cv.circle(blank, (450,450), 40, (0,255,0), thickness = -1)

#4. Draw a line
cv.line(blank, (0,0), (450,450), (0,0,255),thickness = 4)

#5. Add text
cv.putText(blank, "muie pula", (600,600), cv.FONT_HERSHEY_COMPLEX,1.0, (0,0,255))

cv.imshow("Blank",blank)

cv.waitKey(0)