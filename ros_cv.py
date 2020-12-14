#!/usr/bin/python2.7

import rospy
import numpy as np

from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from actuators import PCA9685, PWMSteering, PWMThrottle
from LineFollower import LineFollower
from cv_bridge import CvBridge
import cv2

# car = LineFollower() #generate car class
bridge = CvBridge() #generate image converter

lower = np.array([0, 0, 0])
upper = np.array([20, 255, 255])


def get_steering_and_throttle(mask):
    buf = 0
    threshold = 20
    midpt = len(mask[0]) / 2
    mask = mask[80:,:]
    mask = mask/mask.max()
    left = mask[:, :midpt-buf].sum()
    right = mask[:, buf+midpt:].sum()
    print left, right
    left = left if left > threshold else 0
    right = right if right > threshold else 0
    steering = (left / -1000.) if left > right else (right / 1000.)
    throttle = 0.1
#    throttle = 0 if mask.sum() < threshold else throttle
    return steering, throttle


def callback(msg):
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    steering, throttle = get_steering_and_throttle(mask) 
    throtPub.publish(throttle)
    steerPub.publish(steering) 
    print("Published: \t{:.3f}\t{:.3f}".format(steering, throttle))

rospy.init_node('cv_node')
camSub = rospy.Subscriber('/camera', Image, callback)
throtPub = rospy.Publisher('/throttle', Float32, queue_size=1)
steerPub = rospy.Publisher('/steering', Float32, queue_size=1)
rate = rospy.Rate(10)


while not rospy.is_shutdown():
    rospy.spin()
    rate.sleep()





# while not rospy.is_shutdown():
#   cap = cv2.VideoCapture(0 , cv2.CAP_V4L ) #get live video feed
#   ret, frame = cap.read()
#   resized = cv2.resize(frame,(160,120),interpolation = cv2.INTER_AREA)
#   steering, throttle, composite_img = car.run(resized)
# #   steerPub.publish(car.steering)
# #   throtPub.publish(car.throttle)
#   print("[Steering: ", car.steering , " ] ", " [ Throttle: ", car.throttle," ]")
#   cv2.imshow('composite image', car.cur_image)
#   key = cv2.waitKey(10) & 0xFF
#   if key == ord('q'):
#       break
# cap.release()
# #rate.sleep()

# while not rospy.is_shutdown():
#  cap = cv2.VideoCapture(0 , cv2.CAP_V4L ) #get live video feed
#  ret, frame = cap.read()
#  resized = cv2.resize(frame,(160,120),interpolation = cv2.INTER_AREA)
#  img_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV) # creates ROI while retaining image dimensions
#  mask = cv2.inRange(img_hsv, car.color_thr_low, car.color_thr_hi) # creates image mask
#  # car.run(resized)
#  # steerPub.publish(car.steering)
#  # throtPub.publish(car.throttle)
#  cv2.imshow('frame',mask)
#  key = cv2.waitKey(10) & 0xFF
#  if key == ord('q'):
#      break
#  cap.release()
#  rate.sleep()
