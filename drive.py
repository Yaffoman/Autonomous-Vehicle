#!/usr/bin/env python3

from actuators import PCA9685, PWMSteering, PWMThrottle
from LineFollower import LineFollower
import time
#Config
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


car = LineFollower() #change camera crop & PID settings here

#Main Drive Loop - 4 Nodes: Camera Node + CV Node + Steering Updater Node + Throttle  Node
for i in range (1):
	cur_img = car.take_img() #make this into a publisher node to publish images to topic /video_feed
	steer, throt, _ = car.run(cur_img) #make this into a subscriber node to /video_feed & publish steering & throttle msg to /car_cmd
	print(steer, throt)
	print(car.steering)
	print(car.throttle)
	steering.run(car.steering) #make these into a subscriber to /car_cmd
	throttle.run(car.throttle) #make these into a subscriber to /car_cmd
	time.sleep(.3)
throttle.run(0)
