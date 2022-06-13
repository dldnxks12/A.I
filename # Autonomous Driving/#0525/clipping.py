# https://1ch0.tistory.com/75

import cv2
import sys
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
print("Environment Ready")

# SetUp
pipe = rs.pipeline()
cfg  = rs.config()

pipe_wrapper = rs.pipeline_wrapper(pipe)
pipe_profile = cfg.resolve(pipe_wrapper)

cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()           

clip_distance_in_meters = 0.5 # 0.5m 
clip_distance = clip_distance_in_meters / depth_scale

# Skip 5 first frames
for x in range(5):
    pipe.wait_for_frames()

try:
    while True:

        # Store framesets for processing 
        frameset = pipe.wait_for_frames()
        depth_frame = frameset.get_depth_frame()

        if not depth_frame:
            continue

        depth = np.asanyarray(depth_frame.get_data())                
        
        colorizer = rs.colorizer()
        color_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        cv2.imshow('depth', color_depth)

        depth_3D    = np.dstack((depth, depth, depth))
        rm_depth    = np.where((depth_3D > clip_distance) | (depth_3D <= 0), 0, color_depth)

        cv2.imshow("img", rm_depth)
        cv2.waitKey(1)

        gray = cv2.cvtColor(rm_depth, cv2.COLOR_BGR2GRAY)
        gray[gray != 0] = 255
        # 480 640 
        ROI = gray[240:360]
        #print(gray.shape)
        #print(ROI.shape)
        #sys.exit()
        cv2.imshow("img2", ROI)
        cv2.waitKey(1)

finally:
    pipe.stop()
