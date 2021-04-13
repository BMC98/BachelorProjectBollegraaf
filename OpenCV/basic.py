import cv2 as cv

img = cv.imread('Photos/6.jpeg')

cv.imshow("Picture",img)

#Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

#Blurring
blur = cv.GaussianBlur(img, (11,11), cv.BORDER_DEFAULT)
cv.imshow("Blur",blur)

#Edge cascade
canny = cv.Canny(img, 125,175)
cv.imshow("Edge",canny)

#Dilating the image
dilate = cv.dilate(canny, (3,3), iterations = 5)
cv.imshow("Dilated",dilate)

#Eroding the image
erode = cv.erode(dilate, (3,3), iterations = 5)
cv.imshow("Eroded",erode)

#Resize the image
resize = cv.resize(img, (800,800))
cv.imshow("Resized",resize)


cv.waitKey(0)