import cv2
import sys
import glob
import math
import time
import pickle
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs


# GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get points 
obj_file   = open("objpoints.pkl", "rb")
img_file   = open("imgpoints.pkl", "rb")
rvecs_file = open("rvecs.pkl", "rb")
tvecs_file = open("tvecs.pkl", "rb")
objpoints = pickle.load(obj_file)
imgpoints = pickle.load(img_file)

# Get trained_data 
ret = 3.2668594862688822
dist = np.load("dist_file.npy")
mtx = np.load("mtx_file.npy")
rvecs = pickle.load(rvecs_file)
tvecs = pickle.load(tvecs_file)

obj_file.close()
img_file.close()
rvecs_file.close()
tvecs_file.close()

test_img = cv2.imread("test6.png")
test_img2 = cv2.resize(test_img, (640, 480))

try:
    while True:        
        h, w = test_img.shape[:2]
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue    

        # Convert images to numpy arrays
        img = np.asanyarray(color_frame.get_data())            
        newcameramtx , roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h , x:x+w]
        
        cv2.imshow('RealSense1', img)
        cv2.imshow('RealSense2', dst)

        cv2.waitKey(1)        
        tot_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error
        print("total error: ", tot_error/len(objpoints))

finally:
    # Stop streaming
    pipeline.stop()





