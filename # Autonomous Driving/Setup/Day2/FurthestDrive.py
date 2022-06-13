import math 
import numpy as np
from sensor_msgs.msg import LaserScan # Lidar Data

class FurthestDrive:
    def process_lidar(self, ranges):
        
        NUM_RANGE = len(ranges)                # Range of Lidar Scan : 360 degree
        ANGLE_BETWEEN = 2*np.pi / NUM_RANGE   # 2PI / 360 = 1 radian 
        #ANGLE_BETWEEN = 360 / NUM_RANGE        # 360 / 360 = 1 degree  

        NUM_PER_QUADRANT = NUM_RANGE // 4     # Divide lidar range into 4 parts

        #Find the furthest index at 2, 3 portion 
        #index = np.argmax(ranges[NUM_PER_QUADRANT + 45 : - (NUM_PER_QUADRANT + 45)]) # 135 ~ 225 degree (90 range)
        index = np.argmax(ranges[NUM_PER_QUADRANT + 30 : - (NUM_PER_QUADRANT + 30)]) # 120 ~ 240 degree (120 range)
        # index = np.argmax(ranges[NUM_PER_QUADRANT : - NUM_PER_QUADRANT]) # 90 ~ 270 (180 range)
        #index = np.argmax(ranges[:]) # 0 ~ 360 
        max_index = NUM_PER_QUADRANT + index

        # get steering angle 
        '''
            forward = 0 degree
            left part = negative degree
            right part = positive degree

            ex max_index = 10 -> steering angle = -170 degree
        '''
        steering_angle = max_index * ANGLE_BETWEEN - (NUM_RANGE // 2) * ANGLE_BETWEEN
        speed = 0.1        

        # steering_angle : rad 
        # speed : m/s
        return speed, steering_angle

