#!/usr/bin/python2.7

import rospy
import time
import numpy as np

from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
bridge = CvBridge() #generate image converter

rospy.init_node('cam_node')
camPub = rospy.Publisher('/camera', Image, queue_size=1)
rate = rospy.Rate(30)

cap = cv2.VideoCapture(cv2.CAP_V4L ) #get live video feed

while not rospy.is_shutdown():
  ret, frame = cap.read()
  resized = cv2.resize(frame,(160,120),interpolation = cv2.INTER_AREA)
  cam_img = np.uint8(resized)
  ros_img = bridge.cv2_to_imgmsg(cam_img, "bgr8")
  camPub.publish(ros_img)
#  cv2.imshow('frame',resized)
#  key = cv2.waitKey(10) & 0xFF
#  if key == ord('q'):
#      break
#
  rate.sleep()

cap.release()
