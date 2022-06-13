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
        self.Driver = Driver.FollowTheGap()

        # Call time_callback function for each time_period 
        self.time_period = 0.1
        self.timer = self.create_timer(self.time_period, self.timer_callback)        

        self.best_index = 0
        self.modified_angle = 0.0
        self.steering_angle = 0.0
        self.speed = 0.0
        self.angle = 0.0        

    def get_speed_angle(self, ranges, angle_increment):
        speed, steering_angle, best_index = self.Driver.process_lidar(ranges, angle_increment)
        self.best_index = best_index
        return speed, steering_angle

    # Modifing ... 
    def set_speed_angle(self, speed, steering_angle):                
        self.steering_angle = steering_angle        
        self.speed = speed
        if steering_angle < 3.14: # Turn Left 
            self.angle = 1.0
            self.modified_angle = abs(steering_angle)
            self.time_period = self.modified_angle / 1.0
        else: # Turn Right 
            self.angle = -1.0
            self.modified_angle = abs(steering_angle - 6.28) 
            self.time_period = self.modified_angle / 1.0
        
    def timer_callback(self):
        msg = Twist()
        msg.linear.x  = self.speed # m/s
        msg.angular.z = self.angle # rad/s

        # Publish Message !
        self.publisher.publish(msg)

        # Logging 
        self.get_logger().info("Linear_vel              : %s" % msg.linear.x)
        self.get_logger().info("Angular_vel             : %s" % msg.angular.z)
        self.get_logger().info("Time_period             : %s" % self.time_period)
        self.get_logger().info("Best Index              : %s" % self.best_index)
        #self.get_logger().info("Steering_angle [rad]    : %s" % self.steering_angle)
        #self.get_logger().info("Modified Angle [rad]    : %s" % self.modified_angle)        

class ClientSubscriber(Node):
    def __init__(self):
        super().__init__('my_controller_sub')        
        self.subscriber_lidar = self.create_subscription(LaserScan, '/scan', self.listener_lidar_callback, 10)
        self.lidar_data = 0.0
        self.angle_increment = 0.0

    def listener_lidar_callback(self, msg):
        # inf distance to 1000 
        for i in range(len(msg.ranges)):
            if  str(msg.ranges[i]) == 'inf':
                msg.ranges[i] = 10.0 

        # Processed Lidar Data
        self.lidar_data = msg.ranges
        self.angle_increment = msg.angle_increment

    def return_lidar_data(self):
        return self.lidar_data, self.angle_increment
        

        


