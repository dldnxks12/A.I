import os   # Terminal Control
import sys  # Terminal Control 
import cv2
import math
import time
import pickle
import logging    # For Logger
import subprocess # For parallel Processing 
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs
from model.pspunet import pspunet
from data_loader.display import create_mask


# Get Robot Position (Area = 1/2 * width * height)
def cal_dist(x1, y1, x2, y2, centerX, centerY): 
	Triangle_Area = abs( (x1-centerX)*(y2-centerY) - (y1-centerY)*(x2-centerX) )
	line_distance = math.dist((x1,y1), (x2, y2))
	distance = (2*Triangle_Area) / line_distance 
	return distance

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

IMG_WIDTH  = 480
IMG_HEIGHT = 272
n_classes  = 7

# Camera Parameters
obj_file   = open("objpoints.pkl", "rb")
img_file   = open("imgpoints.pkl", "rb")
rvecs_file = open("rvecs.pkl", "rb")
tvecs_file = open("tvecs.pkl", "rb")
objpoints = pickle.load(obj_file)
imgpoints = pickle.load(img_file)

ret = 3.2668594862688822
dist = np.load("dist_file.npy")
mtx = np.load("mtx_file.npy")
rvecs = pickle.load(rvecs_file)
tvecs = pickle.load(tvecs_file)

obj_file.close()
img_file.close()
rvecs_file.close()
tvecs_file.close()

# Record Video
out = cv2.VideoWriter('0609.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (IMG_WIDTH, IMG_HEIGHT))

# Load trained Model
model = pspunet((IMG_HEIGHT, IMG_WIDTH ,3), n_classes)
model.load_weights("pspunet_weight.h5")

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler('log.txt', mode = 'w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ----  Global Vars ---- #
# Temps
Line_temps       = None
Right_pos_temp   = []    # List for Position Moving Average
Left_pos_temp    = []    # List for Position Moving Average

# Flags
Left_line_flag        = 0 
Right_line_flag       = 0 
No_line_flag          = 0
Pos_flag              = 0  
Init_Subprocess_flag  = True

# Radian
R = 0.0

# Angle
Left_Angle  = 0
Right_Angle = 0
Base_angle  = 35    # ------------------------------ Modifying....

# Position
Right_Avg_Pos  = 0  # Average Pos
Right_Avg_Pos  = 0  # Average Pos
Robot_position = 0  # Current Pos 

# Speed
Base_weight = 100
Left_Wheel  = Base_weight
Right_Wheel = Base_weight

# ---------------------- #

test_img = cv2.imread("shape_img.png")

try:
	while True:
		h, w = test_img.shape[:2]
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		if not color_frame:
		  continue

		# Convert images to numpy arrays
		color_image = np.asanyarray(color_frame.get_data())

		# ----------------- Calibration ---------------------- #
		newcameramtx , roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
		color_image = cv2.undistort(color_image, mtx, dist, None, newcameramtx)
		x, y, w, h = roi
		color_image = color_image[y:y+h , x:x+w]
		# ---------------------------------------------------- #

		# ------------------- Segmentation ------------------- #
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
		# ---------------------------------------------------- #

		video  = np.uint8(frame2 * 255)
		dst    = video[0].copy()
		canvas = dst.copy() * 0
		gray   = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
		height = IMG_HEIGHT // 2

		# Copy Half Size
		lower_gray = gray[:][height-50:]
		canny      = cv2.Canny(lower_gray, 50, 200, None, 3)
		lines      = cv2.HoughLinesP(canny, 1, np.pi/180, 50, None, 50, 10)

		if lines is not None:
			Line_temps   = lines
			No_line_flag = 0
			for i in lines:
				Right_line_flag   = 0
				Left_line_flag    = 0
				pt11        = i[0][0] # x1
				pt12        = i[0][1] # y1
				pt21        = i[0][2] # x2
				pt22        = i[0][3] # y2

				if (pt11 - pt21) != 0: # Not 90 degree
					#    (y1 - y2)    / (x2 - x1)
					R = (pt12 - pt22) / (pt21 - pt11)
					if R < 0: # Got Right-side line 
						R*=-1  
						Right_line_flag   = 1
						Left_line_flag    = 0
					else:     # Got Left-side line 
						Left_line_flag    = 1
						Right_line_flag   = 0

				R = math.atan(R)    # radian ...
				#if R < 0.5236 / 2: # 30 degree
				if R < 0.3 / 2:
					continue

				else:
					Degree = R * (180 / math.pi) # rad to degree -> Current Line Angle 
					print("# ----------- Degree / Direction -------------- #")
					print(f"Degree : {Degree} || Line on Right : {Right_line_flag} || Line on Left : {Left_line_flag}")

					Diff = (Base_angle - Degree) * 3 # --------------------------- Modifying ... 
					#print("DIff", Diff)

					# Get Position (Vertical distance)
					Robot_position = cal_dist(pt11, pt12, pt21, pt22, IMG_WIDTH // 2, IMG_HEIGHT)

					# Line on Left
					if Left_line_flag == 1:
						if Diff <= 0: # go Right
							Left_Wheel  = Base_weight
							Right_Wheel = int(Base_weight - (abs(Diff)))
						else:	      # go Left
							Left_Wheel  = int(Base_weight - (abs(Diff)))
							Right_Wheel = Base_weight

						# Position - Moving Average
						if len(Left_pos_temp) <= 5: # Until Position data > 5
							if len(Left_pos_temp) < 2:
								Left_pos_temp.append(Robot_position)
							else:
								if (abs(Left_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3: # If gap > 20 under 5 times ...
									Pos_flag += 1
								else:			
									Pos_flag = 0					
									Left_pos_temp.append(Robot_position)
							Left_Avg_Pos = np.mean(Left_pos_temp)      # Average all elements in list
						else:							
							if (abs(Left_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3:
								Pos_flag += 1
							else:
								Pos_flag  = 0
								Left_pos_temp.append(Robot_position)	
							Left_Avg_Pos = np.mean(Left_pos_temp[-5:]) # Average last 5 elements in list 							

						# Draw Segmented Line and Put text on Image
						cv2.line(canvas[:][height-50:][:], (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2, cv2.LINE_AA)					
						cv2.putText(canvas, "Left Distance :" + str(Robot_position), (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
						cv2.putText(canvas, "Avg Distance :" + str(Left_Avg_Pos), (50,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)																

						print("# ------------ Pos ------------ # ")
						print("Estimated Robot Position      : " , Robot_position)
						print("Averaged  Robot Position(Left): "  , Left_Avg_Pos)

					# Line on Right
					elif Right_line_flag == 1:
						if Diff <= 0: # go Right
							Left_Wheel  = int(Base_weight - (abs(Diff)))
							Right_Wheel = Base_weight
						else:	      # go Left
							Left_Wheel  = Base_weight
							Right_Wheel = int(Base_weight - (abs(Diff)))

						# Position - Moving Average
						if len(Right_pos_temp) <= 5: # Until Position data > 5
							if len(Right_pos_temp) < 2:
								Right_pos_temp.append(Robot_position)
							else:
								if (abs(Right_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3: # If gap > 20 under 5 times ...
									Pos_flag += 1
								else:			
									Pos_flag = 0					
									Right_pos_temp.append(Robot_position)
							Right_Avg_Pos = np.mean(Right_pos_temp)      # Average all elements in list
						else:							
							if (abs(Right_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3:
								Pos_flag += 1
							else:
								Pos_flag  = 0
								Right_pos_temp.append(Robot_position)	
							Right_Avg_Pos = np.mean(Right_pos_temp[-5:]) # Average last 5 elements in list 

						# Draw Segmented Line and Put text on Image
						cv2.line(canvas[:][height-50:][:], (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2, cv2.LINE_AA)					
						cv2.putText(canvas, "Right Distance :" + str(Robot_position), (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
						cv2.putText(canvas, "Avg Distance :" + str(Right_Avg_Pos), (50,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
						
						print("# ------------ Pos ------------ # ")
						print("Estimated Robot Position        : " , Robot_position)
						print("Averaged  Robot Position (Right): " , Right_Avg_Pos)																					
							
					break # Only 1 line per frame  ...

			# Interpolation 
		elif (lines is None) and (Line_temps is not None):			
			No_line_flag += 1
			if No_line_flag >= 5:            # if not segmented over 5 times ... 
				Left_Wheel  = Base_weight    # Go straing
				Right_Wheel = Base_weight    # Go straing
			else:			
				#print(" No Line Flag : ", No_line_flag)
				for i in Line_temps:
					Right_line_flag   = 0
					Left_line_flag    = 0
					pt11        = i[0][0] # x1
					pt12        = i[0][1] # y1
					pt21        = i[0][2] # x2
					pt22        = i[0][3] # y2

					if (pt11 - pt21) != 0: # Not 90 degree
						#    (y1 - y2)    / (x2 - x1)
						R = (pt12 - pt22) / (pt21 - pt11)
						if R < 0: # Got Right-side line 
							R*=-1  
							Right_line_flag   = 1
							Left_line_flag    = 0
						else:     # Got Left-side line 
							Left_line_flag    = 1
							Right_line_flag   = 0

					R = math.atan(R)    # radian ...
					#if R < 0.5236 / 2: # 30 degree
					if R < 0.3 / 2:
						continue

					else:
						Degree = R * (180 / math.pi)     # rad to degree -> Current Line Angle 
						print("# ----------- Degree / Direction -------------- #")
						print(f"Degree : {Degree} || Line on Right : {Right_line_flag} || Line on Left : {Left_line_flag}")

						Diff = (Base_angle - Degree) * 3 # --------------------------- Modifying ... 
						#print("DIff", Diff)

						# Get Position (Vertical distance)
						Robot_position = cal_dist(pt11, pt12, pt21, pt22, IMG_WIDTH // 2, IMG_HEIGHT)

						# Line on Left
						if Left_line_flag == 1:
							if Diff <= 0: # go Right
								Left_Wheel  = Base_weight
								Right_Wheel = int(Base_weight - (abs(Diff)))
							else:	      # go Left
								Left_Wheel  = int(Base_weight - (abs(Diff)))
								Right_Wheel = Base_weight

							# Position - Moving Average
							if len(Left_pos_temp) <= 5: # Until Position data > 5
								if len(Left_pos_temp) < 2:
									Left_pos_temp.append(Robot_position)
								else:
									if (abs(Left_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3: # If gap > 20 under 5 times ...
										Pos_flag += 1
									else:			
										Pos_flag = 0					
										Left_pos_temp.append(Robot_position)
								Left_Avg_Pos = np.mean(Left_pos_temp)      # Average all elements in list
							else:							
								if (abs(Left_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3:
									Pos_flag += 1
								else:
									Pos_flag  = 0
									Left_pos_temp.append(Robot_position)	
								Left_Avg_Pos = np.mean(Left_pos_temp[-5:]) # Average last 5 elements in list 							

							# Draw Segmented Line and Put text on Image
							cv2.line(canvas[:][height-50:][:], (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2, cv2.LINE_AA)					
							cv2.putText(canvas, "Left Distance :" + str(Robot_position), (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
							cv2.putText(canvas, "Avg Distance :" + str(Left_Avg_Pos), (50,10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)																

							print("# ------------ Pos ------------ # ")
							print("Estimated Robot Position      : " , Robot_position)
							print("Averaged  Robot Position(Left): "  , Left_Avg_Pos)

						# Line on Right
						elif Right_line_flag == 1:
							if Diff <= 0: # go Right
								Left_Wheel  = int(Base_weight - (abs(Diff)))
								Right_Wheel = Base_weight
							else:	      # go Left
								Left_Wheel  = Base_weight
								Right_Wheel = int(Base_weight - (abs(Diff)))

							# Position - Moving Average
							if len(Right_pos_temp) <= 5: # Until Position data > 5
								if len(Right_pos_temp) < 2:
									Right_pos_temp.append(Robot_position)
								else:
									if (abs(Right_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3: # If gap > 20 under 5 times ...
										Pos_flag += 1
									else:			
										Pos_flag = 0					
										Right_pos_temp.append(Robot_position)
								Right_Avg_Pos = np.mean(Right_pos_temp)      # Average all elements in list
							else:							
								if (abs(Right_pos_temp[-1] - Robot_position) > 20) and Pos_flag < 3:
									Pos_flag += 1
								else:
									Pos_flag  = 0
									Right_pos_temp.append(Robot_position)	
								Right_Avg_Pos = np.mean(Right_pos_temp[-5:]) # Average last 5 elements in list 

							# Draw Segmented Line and Put text on Image
							cv2.line(canvas[:][height-50:][:], (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2, cv2.LINE_AA)					
							cv2.putText(canvas, "Right Distance :" + str(Robot_position), (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
							cv2.putText(canvas, "Avg Distance :" + str(Right_Avg_Pos), (50,10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
							
							print("# ------------ Pos ------------ # ")
							print("Estimated Robot Position        : " , Robot_position)
							print("Averaged  Robot Position (Right): " , Right_Avg_Pos)																					
								
						break # Only 1 line per frame  ...

		# Middle Line
		cv2.line(canvas, (240, 272), (240, 136), (0,0,255), 1, cv2.LINE_AA)

		# Record Video
		out.write(canvas)

		# Sending form edit and set start flag 						
		R *= 100
		R = int(R)
		#print("R", R)

		if R == 0:
			pass
			# R = str("000")
		elif R < 10 and R > 0:
			pass
			# R = str("00") + str(R)
		elif R < 100 and R >= 10:
			pass
			# R = str("0") + str(R)

		# TO DO ----------------------------------------------------- #
		logger.info(f"START,0,L,{Left_Wheel},D,1,R,{Right_Wheel},D,1,Z")
		# TO DO ----------------------------------------------------- #
		
		# Reset Radius
		R = 0.0

		#Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', color_image)
		cv2.imshow('Segmented', dst)
		cv2.imshow('canvas', canvas)
		cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
