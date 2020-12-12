from actuators import PCA9685, PWMSteering, PWMThrottle
from LineFollower import LineFollower
import rospy
from std_msgs.msg import Float32
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


rospy.init_node('motor_publisher')
throtPub = rospy.Publisher('/throttle', Float32, queue_size=1)
steerPub = rospy.Publisher('/steering', Float32, queue_size=1)
rate = rospy.Rate(2)


def setSteering(msg):
    steering.run(msg.data)


def setThrottle(msg):
    throttle.run(msg.data)


rospy.init_node('motor_subscriber')
steerSub = rospy.Subscriber('/steering', Float32, setThrottle)   # Create a Subscriber object that will listen to the /counter
throtSub = rospy.Subscriber('/throttle', Float32, setSteering)   # Create a Subscriber object that will listen to the /counter


for i in range (10):
    cur_img = car.take_img() #make this into a publisher node to publish images to topic /video_feed
    steer, throt, _ = car.run(cur_img) #make this into a subscriber node to /video_feed & publish steering & throttle msg to /car_cmd
    print(steer, throt)
    steerPub.publish(steer)
    throtPub.publish(throt)
    time.sleep(.3)
throtPub.publish(0)

