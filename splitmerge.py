import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)


blank = np.zeros(img.shape[:2], dtype='uint8')
blue = cv.merge([img[:,:,0], blank, blank])
green = cv.merge([blank, img[:,:,1], blank])
red = cv.merge([blank, blank, img[:,:,2]])

# cv.imshow('Blue', blue)
# cv.imshow('Green', green)  
# cv.imshow('Red', red)

color_merge = cv.merge([img[:,:,0], img[:,:,1], img[:,:,2]])
cv.imshow('Merged', color_merge)

b,g, r = cv.split(img)
# cv.imshow('Blue', b)
# cv.imshow('Green', g)
# cv.imshow('Red', r)

print(img.shape)
print(b.shape, g.shape, r.shape) 

merged = cv.merge([b, g, r])
# cv.imshow('Merged', merged)

cv.waitKey(0)