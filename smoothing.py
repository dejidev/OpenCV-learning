import cv2 as cv
img = cv.imread('Resources/Photos/cats.jpg')
# cv.imshow('Cat', img)


#Averaging
average = cv.blur(img, (9,5))
# cv.imshow('Averaging', average)


#Gaussian Blur
gauss = cv.GaussianBlur(img, (7,7), 0)
# cv.imshow('Gaussian Blur', gauss)


#Median Blur
median = cv.medianBlur(img, 5)
# cv.imshow('Median Blur', median)


#Bilateral Blur
bilateral = cv.bilateralFilter(img, 15, 35, 25)
cv.imshow('Bilateral Blur', bilateral)


cv.waitKey(0)