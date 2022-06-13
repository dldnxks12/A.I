import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist   # Controll Data (Velocity / Anguluar Velocity) 
from sensor_msgs.msg import Image     # Camera Data 
from sensor_msgs.msg import Imu       # IMU Data
from sensor_msgs.msg import LaserScan # Lidar Data

from cv_bridge import CvBridge # ROS Type Image -> OpenCV Type Image

import capstone.FurthestDrive as Driver

class ClientPublisher(Node):
    def __init__(self):
        super().__init__('my_controller_publisher')

        # Create publisher Node 
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)  # Type : Twist, name : /cmd_vel

        # Create Driver Object 
        self.Driver = Driver.FurthestDrive()

        # Call time_callback function for each time_period 
        self.time_period = 0.5
        self.timer = self.create_timer(self.time_period, self.timer_callback)        

        self.steer_angle = 0.0
        self.speed = 0.0
        self.angle = 0.0

    def get_speed_angle(self, ranges):
        speed, angle = self.Driver.process_lidar(ranges)
        return speed, angle

    def set_speed_angle(self, speed, angle):
                
        self.steer_angle = angle
        self.time_period = abs(angle) / 1.0
        self.speed = speed

        if angle > 0:
            self.angle = 1.0 # 1 rad/s
        else:
            self.angle = -1.0 # 1 rad/s

    def timer_callback(self):

        # return linear vel
        # return angular vel 
        msg = Twist()
        msg.linear.x  = self.speed # m/s
        msg.angular.z = self.angle # rad/s

        # Publish Message !
        self.publisher.publish(msg)

        print("Speed, Angle : ", self.speed, self.angle)
        # Logging 
        self.get_logger().info("Linear_vel     : %s" % msg.linear.x)
        self.get_logger().info("Angular_vel    : %s" % msg.angular.z)
        self.get_logger().info("Steering_angle : %s" % self.steer_angle)
        self.get_logger().info("Time_period    : %s" % self.time_period)


class ClientSubscriber(Node):
    def __init__(self):
        super().__init__('my_controller_sub')        
        self.subscriber_lidar = self.create_subscription(LaserScan, '/scan', self.listener_lidar_callback, 10)
        self.lidar_data = 0.0

    def listener_lidar_callback(self, msg):
        # inf distance to 0 
        for i in range(len(msg.ranges)):
            if  str(msg.ranges[i]) == 'inf':
                msg.ranges[i] = 0.0 

        '''
        # Get Lidar info 
        
        self.get_logger().info("Lidar Info - angle_min [rad] : %s" % msg.angle_min) --- 0 rad (= 0 degree)
        self.get_logger().info("Lidar Info - angle_max [rad] : %s" % msg.angle_max) --- 6.28 rad (= 360 degree)
        self.get_logger().info("Lidar Info - angle_increment [rad] : %s" % msg.angle_increment) --- 0.0174 rad (= 1 degree)

        '''

        # Processed Lidar Data
        self.lidar_data = msg.ranges

    def return_lidar_data(self):
        return self.lidar_data
        

        


