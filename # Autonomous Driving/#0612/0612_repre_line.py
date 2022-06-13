# url https://m.blog.naver.com/windowsub0406/220894645729

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

def ROI(img, vertices, color3 = (255, 255, 255), color1 = 255):
	mask = np.zeros_like(img)
	if len(img.shape) > 2: # 3 channel image
		color = color3
	else:
		color = color1
        	
	cv2.fillPoly(mask, vertices, color)
	ROI_IMG = cv2.bitwise_and(img, mask)
	return ROI_IMG

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10): # 대표선 그리기
        cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

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
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = frame[tf.newaxis, ...]
		frame = frame / 255
		pre   = model.predict(frame)
		pre   = create_mask(pre).numpy()

		frame2 = frame/2
		frame2[0] *= 0
		frame2[0][(pre==6).all(axis=2)] += [1.0, 1.0, 1.0]
		
		# ---------------------------------------------------- #

		video  =  np.uint8(frame2 * 255)		
		dst    = video[0].copy()
		canvas = dst.copy() * 0
		gray   = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

		height = IMG_HEIGHT // 2		

		# Make ROI Image
		# GET ROI
		canny         = cv2.Canny(gray, 50, 200, None, 3)
		height, width = canvas.shape[:2]
		vertices = np.array([[(0,height),(width/2-120, height/2 - 30), (width/2+120, height/2 - 30), (width,height)]], dtype=np.int32)  
		ROI_IMG = ROI(canny, vertices)

		height, width = color_image.shape[:2]				
		vertices = np.array([[(0,height),(width/2-120, height/2 - 30), (width/2+120, height/2 - 30), (width,height)]], dtype=np.int32)  
		color_ROI = ROI(color_image, vertices)        

		# https://deep-learning-study.tistory.com/212 - houghlineP
		#lines      = cv2.HoughLinesP(canny, 1, np.pi/180, 50, None, 50, 10)
		lines      = cv2.HoughLinesP(ROI_IMG, 1, np.pi/180, 50, None, 30, 20)	        

		if lines is not None:
			No_line_flag = 0            
			lines2 = np.squeeze(lines, 1)
			slope_degree = (np.arctan2(lines2[:,1] - lines2[:,3], lines2[:,0] - lines2[:,2]) * 180) / np.pi
		
			# 수평 기울기 제한
			lines2 = lines2[np.abs(slope_degree)<170]
			slope_degree = slope_degree[np.abs(slope_degree)<170]
			# 수직 기울기 제한
			lines2 = lines2[np.abs(slope_degree)>65]
			slope_degree = slope_degree[np.abs(slope_degree)>65]
			
			print(lines2)
			print(slope_degree)
			L_lines = lines2[(slope_degree) > 0, :]			
			R_lines = lines2[(slope_degree) < 0, :]

			print(L_lines)
			print(R_lines)
			
			temp = np.zeros((canvas.shape[0], canvas.shape[1], 3), dtype = np.uint8)			
			L_lines = L_lines[:,None]
			R_lines = R_lines[:,None]			
			if len(L_lines) > 0:
				left_fit_line  = get_fitline(ROI_IMG, L_lines)
				cv2.line(temp, (left_fit_line[0], left_fit_line[1]), (left_fit_line[2], left_fit_line[3]), (255,255,255), cv2.LINE_AA)				
			if len(R_lines) > 0:
				right_fit_line = get_fitline(ROI_IMG, R_lines)			
				cv2.line(temp, (right_fit_line[0], right_fit_line[1]), (right_fit_line[2], right_fit_line[3]), (255,255,255), cv2.LINE_AA)							
			# Draw			
			cv2.imshow("temp", temp)			
		

        
		# Middle Line
		#cv2.line(temp, (240, 272), (240, 136), (0,0,255), 1, cv2.LINE_AA)


		#Show images					
		cv2.imshow("Color", color_image)		
		cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
