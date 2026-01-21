import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
# cv.imshow('Cats', img)



gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)


# Simple Thresholding
ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)


ret, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Thresh Inverse', thresh_inv)



# Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C , cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Mean Threshold', adaptive_thresh)

adaptive_gauss = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Gaussian Threshold', adaptive_gauss)


cv.waitKey(0)