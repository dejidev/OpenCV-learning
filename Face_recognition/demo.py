print("Hello, World!")

import cv2 as cv

img = cv.imread('Resources/Photos/cat.jpg')

# cv.imshow('Cat', img)

#Convert to greyscale
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Grey', grey)

#Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)


#Edge cascade
cany = cv.Canny(img, 125 ,175)
# cv.imshow('Canny Edges', cany)


#Dialate the image  
dilated = cv.dilate(cany, (7,7), iterations=3)
# cv.imshow('Dilated', dilated)


#Broading
eroded = cv.erode(dilated, (7,7), iterations=3)
# cv.imshow('Eroded', eroded)


#Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized', resized)


#Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)



cv.waitKey(0)
