import cv2 as cv
import numpy as np

img = cv.imread("7-images/football.jpg")
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

def gradient_magnitude(img):
    resx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)  
    resy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    res = np.sqrt(resx**2 + resy**2)  
    return np.uint8(255 * res / np.max(res))

def gradient_orientation(img):
    resx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)  
    resy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    orientation = np.arctan2(resy, resx) 
    orientation_normalized = cv.normalize(orientation, None, 0, 255, cv.NORM_MINMAX)
    return orientation_normalized.astype(np.uint8)

gm = gradient_magnitude(img)
_, thresholded = cv.threshold(gm, 100, 255, cv.THRESH_BINARY)
orientation=gradient_orientation(img)

cv.imshow("img", img)
cv.imshow("Gradient Magnitude", gm)
cv.imshow("Thresholded", thresholded)
cv.imshow("Gradient orientation", orientation)
cv.waitKey(0)
cv.destroyAllWindows()
