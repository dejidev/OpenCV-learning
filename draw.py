import cv2 as cv
import numpy as np


blank = np.zeros((500, 500, 3), dtype='uint8')

# cv.imshow('Blank', blank)

# img = cv.imread('Resources/Photos/cat.jpg')
# cv.imshow('Cat', img)


#Paint the image a certain color
# blank[:] = 0,0,255
# cv.imshow('Red', blank)


# Draw a rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
# cv.imshow('Rectangle', blank)


# Draw a circle
cv.circle(blank, (blank.shape[1]//2 , blank.shape[0]//2), 40, (0,0,255), thickness=-1)
# cv.imshow('Circle', blank)


#Draw a line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=3)
# cv.imshow('Line', blank)


# Write text
cv.putText(blank, 'Hello World', (225, 225), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
cv.imshow('Text', blank)

cv.waitKey(0)