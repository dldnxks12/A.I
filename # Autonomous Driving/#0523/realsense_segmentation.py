import cv2
import sys
import time
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs
from model.pspunet import pspunet
from data_loader.display import create_mask

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
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

IMG_WIDTH  = 480
IMG_HEIGHT = 272
n_classes  = 7

# Record Video
out = cv2.VideoWriter('0523.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (IMG_WIDTH, IMG_HEIGHT))

# Load trained Model
model = pspunet((IMG_HEIGHT, IMG_WIDTH ,3), n_classes)
model.load_weights("pspunet_weight.h5")

try:
	while True:
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		if not color_frame:
		    continue

		# Convert images to numpy arrays
		color_image = np.asanyarray(color_frame.get_data())
		frame = cv2.resize(color_image, (IMG_WIDTH, IMG_HEIGHT))
		frame = cv2.resize(color_image,(IMG_WIDTH, IMG_HEIGHT))

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = frame[tf.newaxis, ...]
		frame = frame / 255
		pre   = model.predict(frame)
		pre   = create_mask(pre).numpy()

		frame2 = frame/2
		frame2[0] *= 0
		frame2[0][(pre==6).all(axis=2)] += [1.0, 1.0, 1.0]

		video  = np.uint8(frame2 * 255)		
		dst    = video[0].copy()
		canvas = dst.copy() * 0
		gray   = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
		height = IMG_HEIGHT // 2
		
		# Copy Half Size
		lower_gray = gray[:][height:]
		canny      = cv2.Canny(lower_gray, 50, 200, None, 3) 
		lines      = cv2.HoughLinesP(canny, 1, np.pi/180, 50, None, 50, 10)		
	
		if lines is not None:
			for i in lines:
				pt11 = i[0][0]
				pt12 = i[0][1]
				pt21 = i[0][2]
				pt22 = i[0][3]

				if (pt11 - pt21) != 0:
					R = abs(pt12 - pt22) / abs(pt11 - pt21)

				if R < 0.7:
					continue

				elif R == 1e-3:
					pass
					# Turn
				else:
					cv2.line(canvas[:][height:][:], (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 1, cv2.LINE_AA)
				

		# Middle Line
		cv2.line(canvas, (240, 272), (240, 136), (255,255,255), 1, cv2.LINE_AA)
						
		out.write(canvas)

		#Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', color_image)
		cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
    
    
