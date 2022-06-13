############# Used Method ##############

# Camera Calibration
# Side Walk Segmentation 
# Remove noise with Morpholgy - Opening
# Get Contours and select biggest Area Contour
# HoughLinesP
# Get Main Left / Right Line
# Get ROI 
# Interpolation
# Line Moving Average
# Get Angle    + moving average + remove outliar
# Get Distance + moving average + remove outliar

#### Remove ####

	# Canny Edge
	# Bird-eye-view

#### Todo ####

	# Control 
		# Left / Right distance
		# Left / Right angle
	# Communication relate work

########################################

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

# Get Angle
def get_angle(Points):
	angle = (np.arctan2(Points[1] - Points[3], Points[0] - Points[2]) * 180) / np.pi 	
	return angle

# Get ROI
def ROI(img, vertices, color3 = (255, 255, 255), color1 = 255):
	mask = np.zeros_like(img)
	if len(img.shape) > 2: # 3 channel image
		color = color3
	else:
		color = color1
        	
	cv2.fillPoly(mask, vertices, color)
	ROI_IMG = cv2.bitwise_and(img, mask)
	return ROI_IMG

# Get Main line
def get_fitline(img, f_lines): # 대표선 구하기   
    lines = np.squeeze(f_lines, 1)
    lines = lines.reshape(lines.shape[0]*2,2)
    rows,cols = img.shape[:2]
    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    
    result = [x1,y1,x2,y2]
    return result


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
L_x1 = [] # Points temp
L_y1 = [] # Points temp
L_x2 = [] # Points temp
L_y2 = [] # Points temp
R_x1 = [] # Points temp
R_y1 = [] # Points temp
R_x2 = [] # Points temp
R_y2 = [] # Points temp
Left_Avg_points_temp  = 0
Right_Avg_points_temp = 0

# Flags
Left_line_interpolation  = 0
Right_line_interpolation = 0
No_line_flag             = 0
Left_Pos_flag            = 0
Left_Angle_flag          = 0 
Right_Pos_flag           = 0  
Right_Angle_flag          = 0 

# Angle
Base_angle  = 110 
Left_Angle  = 90
Right_Angle = 90

# Position
Left_distance  = 0  # Current Pos 
Right_distance = 0  # Current Pos 

# Control
Left_Difference  = 0 
Right_Difference = 0
Base_weight = 100
Left_Wheel  = Base_weight
Right_Wheel = Base_weight

# Moving Average
Left_pos_temp     = []    # List for Position Moving Average
Left_angle_temp   = []    # List for Angle Moving Average
Right_pos_temp    = []    # List for Position Moving Average
Right_angle_temp  = []    # List for Angle Moving Average
Left_Avg_Pos   = 0        # Average Position
Left_Avg_Ang   = 0        # Average Angle
Right_Avg_Pos  = 0        # Average Position
Right_Avg_Ang  = 0        # Average Angle

# ---------------------- #

test_img = cv2.imread("test.png")

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
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = frame[tf.newaxis, ...]
		frame = frame / 255
		pre   = model.predict(frame)
		pre   = create_mask(pre).numpy()

		frame2 = frame/2
		frame2[0] *= 0
		frame2[0][(pre==6).all(axis=2)] += [1.0, 1.0, 1.0]		
		# ---------------------------------------------------- #

		video   =  np.uint8(frame2 * 255)		
		dst     =  video[0].copy()
		canvas  =  dst.copy() * 0
		canvas2 =  dst.copy() * 0
		gray    =  cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

		height = IMG_HEIGHT // 2		

		# Remove small noise 
		k          = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
		opening    = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)

		_, opening = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)

		# Get Contour of ROI Image (draw external contour only)
		contours, hierachy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) > 0 :
			max_cntr = contours[0]
			max_area = cv2.contourArea(max_cntr)
			for i in contours:				
				if max_area < cv2.contourArea(i):
					max_area =  cv2.contourArea(i)
					max_cntr = i

			# Normal Contours
			# cv2.drawContours(canvas2, [cntr], -1, (0, 255, 0), 1)

			# Simplify Contours
			#epsilon = 0.05 * cv2.arcLength(cntr, True)
			#approx = cv2.approxPolyDP(cntr, epsilon, True)
			#cv2.drawContours(canvas2, [approx], -1, (255, 255, 0), 1)

			# hull convex Contours
			hull = cv2.convexHull(max_cntr)
			cv2.drawContours(canvas2, [hull], -1, (255, 255, 255), 1)

		cv2.imshow("Preprocessed Contour Image", canvas2)

		# Make ROI
		height, width  = canvas2.shape[:2]
		vertices = np.array([[(0,height),(width/2-120, height/2 - 30), (width/2+120, height/2 - 30), (width,height)]], dtype=np.int32)  
		ROI_IMG = ROI(canvas2, vertices)
		ROI_IMG = cv2.cvtColor(ROI_IMG, cv2.COLOR_BGR2GRAY)

		# Make Color ROI
		# height, width = color_image.shape[:2]				
		# vertices = np.array([[(0,height),(width/2-120, height/2 - 30), (width/2+120, height/2 - 30), (width,height)]], dtype=np.int32)  
		# color_ROI = ROI(color_image, vertices)        

		lines      = cv2.HoughLinesP(ROI_IMG, 1, np.pi/180, 50, None, 30, 20)	       

		if lines is not None:
			No_line_flag = 0             

			# Eliminate axis 1
			lines2 = np.squeeze(lines, 1)
			slope_degree = (np.arctan2(lines2[:,1] - lines2[:,3], lines2[:,0] - lines2[:,2]) * 180) / np.pi

			lines2 = lines2[np.abs(slope_degree) < 160]
			slope_degree = slope_degree[np.abs(slope_degree) < 160]			
			lines2 = lines2[np.abs(slope_degree) > 85]
			slope_degree = slope_degree[np.abs(slope_degree) > 85]
									
			L_lines = lines2[(slope_degree) > 0, :]			
			R_lines = lines2[(slope_degree) < 0, :]

			# Restore axis 1
			L_lines = L_lines[:,None]
			R_lines = R_lines[:,None]				

			# Get Main Line of Left Side
			if len(L_lines) > 0:
				Left_line_interpolation = 0 				
				left_fit_line  = get_fitline(ROI_IMG, L_lines)
				
				if len(L_x1) < 3:
					L_x1.append(left_fit_line[0])
					L_y1.append(left_fit_line[1])
					L_x2.append(left_fit_line[2])
					L_y2.append(left_fit_line[3])
					x1 = int(sum(L_x1) / len(L_x1))
					y1 = int(sum(L_y1) / len(L_y1))
					x2 = int(sum(L_x2) / len(L_x2)) 
					y2 = int(sum(L_y2) / len(L_y2)) 
					Left_Avg_points_temp = [x1, y1, x2, y2]

				else:
					L_x1.append(left_fit_line[0])
					L_y1.append(left_fit_line[1])
					L_x2.append(left_fit_line[2])
					L_y2.append(left_fit_line[3])

					x1 = int(sum(L_x1[-3:]) / 3)
					y1 = int(sum(L_y1[-3:]) / 3)
					x2 = int(sum(L_x2[-3:]) / 3)
					y2 = int(sum(L_y2[-3:]) / 3)
					Left_Avg_points_temp = [x1, y1, x2, y2]

				# Get angle and distance
				Left_Angle     = get_angle(left_fit_line)				
				Left_distance = cal_dist(x1, y1, x2, y2, width//2, height)

				# Distance Moving Average + remove outliar 
				if len(Left_pos_temp) <= 3: 
					if len(Left_pos_temp) < 2:
						Left_pos_temp.append(Left_distance) 
					else:
						if (abs(Left_pos_temp[-1] - Left_distance) > 100) and Left_Pos_flag < 3:
							Left_Pos_flag += 1
						else:
							Left_Pos_flag = 0 
							Left_pos_temp.append(Left_distance)
					Left_Avg_Pos = np.mean(Left_pos_temp)						
				else:
					if(abs(Left_pos_temp[-1] - Left_distance) > 100) and Left_Pos_flag < 3:
						Left_Pos_flag += 1
					else:
						Left_Pos_flag = 0 
						Left_pos_temp.append(Left_distance)
					Left_Avg_Pos = np.mean(Left_pos_temp[-3:])

				# Angle Moving Average + remove outliar 
				if len(Left_angle_temp) <= 3: 
					if len(Left_angle_temp) < 2:
						Left_angle_temp.append(Left_Angle) 
					else:
						if (abs(Left_angle_temp[-1] - Left_Angle) > 20) and Left_Angle_flag < 3:
							Left_Angle_flag += 1
						else:
							Left_Angle_flag = 0 
							Left_angle_temp.append(Left_Angle)
					Left_Avg_Ang = np.mean(Left_angle_temp)						
				else:
					if(abs(Left_angle_temp[-1] - Left_Angle) > 20) and Left_Angle_flag < 3:
						Left_Angle_flag += 1
					else:
						Left_Angle_flag = 0 
						Left_angle_temp.append(Left_Angle)
					Left_Avg_Ang = np.mean(Left_angle_temp[-3:])

				# Calc Control input
				Left_Difference = (Base_angle - Left_Avg_Ang)
				if Left_Difference <= 0: # go right
					Left_Wheel  = Base_weight
					Right_Wheel = int(Base_weight - abs(Left_Difference))
				else:
					Left_Wheel  = int(Base_weight - abs(Left_Difference))					
					Right_Wheel = Base_weight

				cv2.line(canvas, (x1, y1), (x2, y2), (255,255,255), 2)								
				cv2.putText(canvas, "Left Angle        :" + str(Left_Angle),    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	        
				cv2.putText(canvas, "Left Avg Angle    :" + str(Left_Avg_Ang),  (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	        				
				cv2.putText(canvas, "Left Distance     :" + str(Left_distance), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)					
				cv2.putText(canvas, "Left Avg Distance :" + str(Left_Avg_Pos),  (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	

			else: # Interpolation
				Left_line_interpolation += 1
				if Left_line_interpolation >= 3:
					Left_Angle     = 90.0           # Go straing 
					Left_Wheel     = Base_weight    # Go straing					
				elif (Left_line_interpolation < 3) and (Left_Avg_points_temp != 0):
					# Left_Angle  = Left_Angle 
					x1, y1, x2, y2 = Left_Avg_points_temp
					Left_distance = cal_dist(x1, y1, x2, y2, width//2, height)
					cv2.line(canvas, (x1, y1), (x2, y2), (255,255,255), 2)	

					# Calc Control input
					Left_Difference = (Base_angle - Left_Avg_Ang)
					if Left_Difference <= 0: # go right
						Left_Wheel  = Base_weight
						Right_Wheel = int(Base_weight - abs(Left_Difference))
					else:
						Left_Wheel  = int(Base_weight - abs(Left_Difference))					
						Right_Wheel = Base_weight
											
					cv2.putText(canvas, "Left Angle        :" + str(Left_Angle),    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	        
					cv2.putText(canvas, "Left Avg Angle    :" + str(Left_Avg_Ang),  (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	        				
					cv2.putText(canvas, "Left Distance     :" + str(Left_distance), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)					
					cv2.putText(canvas, "Left Avg Distance :" + str(Left_Avg_Pos),  (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	

			# Get Main Line of Right Side
			if len(R_lines) > 0:
				Right_line_interpolation = 0 
				right_fit_line  = get_fitline(ROI_IMG, R_lines)

				if len(R_x1) < 3:
					R_x1.append(right_fit_line[0])
					R_y1.append(right_fit_line[1])
					R_x2.append(right_fit_line[2])
					R_y2.append(right_fit_line[3])
					x1 = int(sum(R_x1) / len(R_x1))
					y1 = int(sum(R_y1) / len(R_y1)) 
					x2 = int(sum(R_x2) / len(R_x2))
					y2 = int(sum(R_y2) / len(R_y2)) 
					Right_Avg_points_temp = [x1, y1, x2, y2]

				else:
					R_x1.append(right_fit_line[0])
					R_y1.append(right_fit_line[1])
					R_x2.append(right_fit_line[2])
					R_y2.append(right_fit_line[3])

					x1 = int(sum(R_x1[-3:]) / 3)
					y1 = int(sum(R_y1[-3:]) / 3)
					x2 = int(sum(R_x2[-3:]) / 3)
					y2 = int(sum(R_y2[-3:]) / 3)
					Right_Avg_points_temp = [x1, y1, x2, y2]

				# Get Angle and Distance
				Right_Angle     = get_angle(right_fit_line) 				
				Right_distance = cal_dist(x1, y1, x2, y2, width // 2, height)

				# Distance Moving Average + remove outliar 
				if len(Right_pos_temp) <= 3: 
					if len(Right_pos_temp) < 2:
						Right_pos_temp.append(Right_distance) 
					else:
						if (abs(Right_pos_temp[-1] - Right_distance) > 100) and Right_Pos_flag < 3:
							Right_Pos_flag += 1
						else:
							Right_Pos_flag = 0 
							Right_pos_temp.append(Right_distance)
					Right_Avg_Pos = np.mean(Right_pos_temp)						
				else:
					if(abs(Right_pos_temp[-1] - Right_distance) > 100) and Right_Pos_flag < 3:
						Right_Pos_flag += 1
					else:
						Right_Pos_flag = 0 
						Right_pos_temp.append(Right_distance)
					Right_Avg_Pos = np.mean(Right_pos_temp[-3:])
	
				# Angle Moving Average + remove outliar 
				if len(Right_angle_temp) <= 3: 
					if len(Right_angle_temp) < 2:
						Right_angle_temp.append(Right_Angle) 
					else:
						if (abs(Right_angle_temp[-1] - Right_Angle) > 20) and Right_Angle_flag < 3:
							Right_Angle_flag += 1
						else:
							Right_Angle_flag = 0 
							Right_angle_temp.append(Right_Angle)
					Right_Avg_Ang = np.mean(Right_angle_temp)						
				else:
					if(abs(Right_angle_temp[-1] - Right_Angle) > 20) and Right_Angle_flag < 3:
						Right_Angle_flag += 1
					else:
						Right_Angle_flag = 0 
						Right_angle_temp.append(Right_Angle)
					Right_Avg_Ang = np.mean(Right_angle_temp[-3:])

				# Draw
				cv2.line(canvas, (x1, y1), (x2, y2), (255,255,255), 2)	
				cv2.putText(canvas, "Right Angle        :"    + str(Right_Angle),    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
				cv2.putText(canvas, "Right Avg Angle    :"    + str(Right_Angle),    (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
				cv2.putText(canvas, "Right Distance     :"    + str(Right_distance), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)					
				cv2.putText(canvas, "Right Avg Distance :"    + str(Right_Avg_Pos),  (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	

			else: # Interpolation
				Right_line_interpolation += 1
				if Right_line_interpolation >= 3:
					Right_Angle = 90.0				

				elif (Right_line_interpolation < 3) and (Right_Avg_points_temp != 0):
					# Right_Angle = Right_Angle
					x1, y1, x2, y2 = Right_Avg_points_temp
					Right_distance = cal_dist(x1, y1, x2, y2, width // 2, height)
					cv2.line(canvas, (x1, y1), (x2, y2), (255,255,255), 2)	
					cv2.putText(canvas, "Right Angle        :"    + str(Right_Angle),    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
					cv2.putText(canvas, "Right Avg Angle    :"    + str(Right_Angle),    (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
					cv2.putText(canvas, "Right Distance     :"    + str(Right_distance), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)					
					cv2.putText(canvas, "Right Avg Distance :"    + str(Right_Avg_Pos),  (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	

		# Interpolation
		elif lines is None: 
			No_line_flag += 1
			if No_line_flag >= 3:
				Left_Angle, Right_Angle = 90.0, 90.0
				Left_Wheel, Right_Wheel = Base_weight, Base_weight

			else:    # Interpolation
				if Left_Avg_points_temp != 0:
					x1, y1, x2, y2 = Left_Avg_points_temp
					Left_angle = get_angle([x1, y1, x2, y2])
					Left_distance = cal_dist(x1, y1, x2, y2, width // 2, height)	

					# Calc Control input
					Left_Difference = (Base_angle - Left_Avg_Ang)
					if Left_Difference <= 0: # go right
						Left_Wheel  = Base_weight
						Right_Wheel = int(Base_weight - abs(Left_Difference))
					else:
						Left_Wheel  = int(Base_weight - abs(Left_Difference))					
						Right_Wheel = Base_weight

					cv2.line(canvas, (x1, y1), (x2, y2), (255,255,255), 2)	
					cv2.putText(canvas, "Left Angle        :" + str(Left_Angle),    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	        
					cv2.putText(canvas, "Left Avg Angle    :" + str(Left_Avg_Ang),  (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	        				
					cv2.putText(canvas, "Left Distance     :" + str(Left_distance), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)					
					cv2.putText(canvas, "Left Avg Distance :" + str(Left_Avg_Pos),  (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	

				if Right_Avg_points_temp != 0:
					x1, y1, x2, y2 = Right_Avg_points_temp
					Right_Angle = get_angle([x1, y1, x2, y2])
					Right_distance = cal_dist(x1, y1, x2, y2, width // 2, height)				
														
					cv2.line(canvas, (x1, y1), (x2, y2), (255,255,255), 2)	
					cv2.putText(canvas, "Right Angle        :"    + str(Right_Angle),    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
					cv2.putText(canvas, "Right Avg Angle    :"    + str(Right_Angle),    (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	
					cv2.putText(canvas, "Right Distance     :"    + str(Right_distance), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)					
					cv2.putText(canvas, "Right Avg Distance :"    + str(Right_Avg_Pos),  (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)	

		# Middle Line
		cv2.line(canvas, (240, 272), (240, 136), (0,0,255), 1, cv2.LINE_AA)

		#Show images									
		cv2.imshow("Color", color_image)
		cv2.imshow("ROI_IMG", ROI_IMG)		
		cv2.imshow("Detected Lines", canvas)

		cv2.waitKey(1)

		print(f" L : {Left_Wheel} R : {Right_Wheel}")
		# UART
		#logger.info(f"START,0,L,{Left_Wheel},D,1,R,{Right_Wheel},D,1,Z")
		
	
finally:
    # Stop streaming
    pipeline.stop()
