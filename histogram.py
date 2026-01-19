import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  


#GrayScale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
# print(gray_hist)

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()



#Historam of a masked image
blank = np.zeros(img.shape[:2], dtype='uint8')
circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
masked = cv.bitwise_and(img, img, mask=circle)
cv.imshow('Masked Image', masked)

# gray_masked = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
# gray_masked_hist = cv.calcHist([gray_masked], [0], circle, [256], [0, 256])

# plt.figure()
# plt.title('Color Histogram with Masking')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.plot(gray_masked_hist)
# plt.xlim([0, 256])
# plt.show()




plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()



cv.waitKey(0)