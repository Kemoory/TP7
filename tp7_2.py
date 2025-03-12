import cv2
import numpy as np

coins_img = cv2.imread('7-images/coins.png', cv2.IMREAD_COLOR)
corridor_img = cv2.imread('7-images/corridor.png', cv2.IMREAD_COLOR)

gray_corridor = cv2.cvtColor(corridor_img, cv2.COLOR_BGR2GRAY)
edges_corridor = cv2.Canny(gray_corridor, 50, 150)

lines = cv2.HoughLinesP(edges_corridor, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(corridor_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

gray_coins = cv2.cvtColor(coins_img, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.medianBlur(gray_coins, 5)

circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        cv2.circle(coins_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        cv2.circle(coins_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)

cv2.imshow('Detected Lines in Corridor', corridor_img)
cv2.imshow('Detected Circles in Coins', coins_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
