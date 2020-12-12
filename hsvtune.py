#!/usr/bin/env python

import cv2
import numpy as np


def callback(x):
    pass


cap = cv2.VideoCapture(cv2.CAP_V4L)

cv2.namedWindow('sliders')

lowH = 0
highH = 179
lowS = 0
highS = 255
lowV = 0
highV = 255

cv2.createTrackbar('lowH', 'sliders', lowH, 179, callback)
cv2.createTrackbar('highH', 'sliders', highH, 179, callback)

cv2.createTrackbar('lowS', 'sliders', lowS, 255, callback)
cv2.createTrackbar('highS', 'sliders', highS, 255, callback)

cv2.createTrackbar('lowV', 'sliders', lowV, 255, callback)
cv2.createTrackbar('highV', 'sliders', highV, 255, callback)

# cv2.setTrackbarPos('lowH','sliders',15)
# cv2.setTrackbarPos('highH','sliders',50)
# cv2.setTrackbarPos('lowS','sliders',0)
# cv2.setTrackbarPos('highS','sliders',160)
# cv2.setTrackbarPos('lowV','sliders',160)
# cv2.setTrackbarPos('highV','sliders',255)

cv2.setTrackbarPos('lowH','sliders',24) # TENNIS BALL
cv2.setTrackbarPos('highH','sliders',85)
cv2.setTrackbarPos('lowS','sliders',55)
cv2.setTrackbarPos('highS','sliders',255)
cv2.setTrackbarPos('lowV','sliders',50)
cv2.setTrackbarPos('highV','sliders',255)


while True:
    ret, frame = cap.read()
    #frame = cv2.imread('image.png')
    frame = cv2.resize(frame,(160,120),interpolation = cv2.INTER_AREA)
    # get trackbar positions
    lowH = cv2.getTrackbarPos('lowH', 'sliders')
    highH = cv2.getTrackbarPos('highH', 'sliders')
    lowS = cv2.getTrackbarPos('lowS', 'sliders')
    highS = cv2.getTrackbarPos('highS', 'sliders')
    lowV = cv2.getTrackbarPos('lowV', 'sliders')
    highV = cv2.getTrackbarPos('highV', 'sliders')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([lowH, lowS, lowV])
    higher = np.array([highH, highS, highV])
    mask = cv2.inRange(hsv, lower, higher)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
