import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist   # Controll Data (Velocity / Anguluar Velocity) 
from sensor_msgs.msg import Image     # Camera Data 
from sensor_msgs.msg import Imu       # IMU Data
from sensor_msgs.msg import LaserScan # Lidar Data

from cv_bridge import CvBridge # ROS Type Image -> OpenCV Type Image

class ClientPublisher(Node):
    def __init__(self):
        super().__init__('my_controller_publisher')

        # Create publisher Node 
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)  # Type : Twist, name : /cmd_vel
        
        # Call time_callback function for each time_period 
        time_period = 0.5
        self.timer = self.create_timer(time_period, self.timer_callback)

        self.speed = 0.0
        self.angle = 0.0

    def set_speed_velocity(self, lidar_data):

        self.speed = lidar_data[100]
        self.angle = lidar_data[200]

    def timer_callback(self):
        msg = Twist()
        msg.linear.x  = self.speed
        msg.angular.z = self.angle

        # Publish Message !
        self.publisher.publish(msg)

        # Logging 
        self.get_logger().info("Publishing : %s" % msg)


class ClientSubscriber(Node):
    def __init__(self):
        super().__init__('my_controller_sub')        
        self.subscriber_lidar = self.create_subscription(LaserScan, '/scan', self.listener_lidar_callback, 10)

        self.lidar_data = 0.0

    def listener_lidar_callback(self, msg):

        '''
        print(msg.ranges)
        print(type(msg.ranges))
        '''

        # Time complexity : O(n)         
        # inf distance --> 0 
        for i in range(len(msg.ranges)):
            if  str(msg.ranges[i]) == 'inf':
                msg.ranges[i] = 0.0 

        # print(msg.ranges)

        # msg.ranges : float type 
        # self.get_logger().info('Subscribed lidar data 1 : %s' % msg.ranges[0])

        self.lidar_data = msg.ranges

    def return_lidar_data(self):
        return self.lidar_data
        

        


