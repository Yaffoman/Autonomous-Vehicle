#!/usr/bin/env python

import rospy
import time

from std_msgs.msg import Float32
from actuators import PCA9685, PWMSteering, PWMThrottle
from LineFollower import LineFollower

# Physical Car Parameters Config
addr = 0x40
steering_ch = 0
steering_left = 270
steering_mid = 380
steering_right = 490

throttle_ch = 1
throttle_fwd = 460
throttle_netural = 380
throttle_bwd = 240

#Set Up PCA9685 I2C Connetion
steering_controller = PCA9685(steering_ch, addr, busnum=1)
steering = PWMSteering(controller=steering_controller,
                                left_pulse=steering_left,
                                right_pulse=steering_right)

throttle_controller = PCA9685(throttle_ch, addr, busnum=1)
throttle = PWMThrottle(controller=throttle_controller,
                                max_pulse=throttle_fwd,
                                zero_pulse=throttle_netural,
                                min_pulse=throttle_bwd)


# car = LineFollower() #intialize car class

def setSteering(msg):
    steering.run(msg.data)

def setThrottle(msg):
    throttle.run(msg.data)

rospy.init_node('car_node')
steerSub = rospy.Subscriber('/steering', Float32, setSteering)   # Create a Subscriber object that will listen to the /counter
throtSub = rospy.Subscriber('/throttle', Float32, setThrottle)   # Create a Subscriber object that will listen to the /counter
rate = rospy.Rate(2)

while not rospy.is_shutdown():
    rospy.spin()
    rate.sleep()

# while not rospy.is_shutdown():
#   steerPub.publish(car.steering)
#   throtPub.publish(car.throttle)
#   print "Distance Left: ", left_dis , " meters"
#   rate.sleep()
