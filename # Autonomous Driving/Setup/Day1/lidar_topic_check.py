'''
import rclpy
import capstone.comm as comm

def main():
    rclpy.init(args = None)

    my_controller_subscriber = comm.ClientSubscriber()
    rclpy.spin_once(my_controller_subscriber)  # Once in a while loop

    my_controller_subscriber.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()

'''

import rclpy
import capstone.comm as comm

def main():
    rclpy.init(args = None)

    # Create Publish / Subscriber Object
    my_controller_publisher  = comm.ClientPublisher()
    my_controller_subscriber = comm.ClientSubscriber()

    while(1):
        # Keep Publisher Node working
        rclpy.spin_once(my_controller_publisher)   # Once in a while loop
        rclpy.spin_once(my_controller_subscriber)  # Once in a while loop

        # Get Subscriber
        lidar_data = my_controller_subscriber.return_lidar_data()

        # Process and tossing data 
        my_controller_publisher.set_speed_velocity(lidar_data)

    # Finish
    my_controller_publisher.destroy_node()
    my_controller_subscriber.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()
