#!/usr/bin/env python3

#from actuators import PCA9685, PWMSteering, PWMThrottle
from LineFollower import LineFollower
import time
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
    [(0,130),(160,130),(160,50),(0,50)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons, color = 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


car = LineFollower() #change camera crop & PID settings here
cap = cv2.VideoCapture('new_track_2.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('demo.avi',fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))
if not cap.isOpened():
    print("Error opening Video File.")
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, car.color_thr_low, car.color_thr_hi)
        crop = roi(mask)
        cv2.imshow('results',crop)
        out.write(crop)
        cv2.waitKey(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if frame is read correctly, ret is True
        if not ret:
            print("Can't retrieve frame - stream may have ended. Exiting..")
            break
except:
    print("Video has ended.")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
