#!/usr/bin/python2.7
import numpy as np
import cv2

cap = cv2.VideoCapture(cv2.CAP_V4L ) #get live video feed

lower = np.array([0, 0, 0])
upper = np.array([20, 255, 255])

ret, frame = cap.read()
resized = cv2.resize(frame,(160,120),interpolation = cv2.INTER_AREA)
resized = resized[40:,:]
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
cv2.imwrite('image.png',resized)
cv2.imwrite('mask.png',mask)
#  key = cv2.waitKey(10) & 0xFF
#  if key == ord('q'):
#      break
#

cap.release()
