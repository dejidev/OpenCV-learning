import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cat.jpg')

cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

canny = cv.Canny(gray, 125, 175)
cv.imshow('Canny Edges', canny)

contours, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours found!')

cv.waitKey(0)