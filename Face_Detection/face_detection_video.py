import cv2 as cv
import numpy as np


img = cv.imread('../Resources/Videos/dog.jpg')
# cv.imshow('Group', img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)


#Face detection in video
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
print(f'Number of faces found = {len(faces_rect)}')




cv.waitKey(0)