import cv2
import sys
import time
import tensorflow as tf
from model.pspunet import pspunet
from data_loader.display import create_mask
import numpy as np

# ----------------------------------------------------------------------- #
# GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)
# ----------------------------------------------------------------------- #

# Video Setting
cap = cv2.VideoCapture(0)

if(cap.isOpened() == False):
	print("Camera Error...")

IMG_WIDTH = 480
IMG_HEIGHT = 272
n_classes = 7

# for estimating real position
Camera_WIDTH  = int(cap.get(3))
Camera_HEIGHT = int(cap.get(4))

# Size Ratio
Ratio = Camera_WIDTH / IMG_WIDTH


print("##########################")
print("#------------------------#")
print("Train  Image Shape: ", IMG_WIDTH, IMG_HEIGHT)
print("Camera Image Shape: ", Camera_WIDTH, Camera_HEIGHT)
print("Ratio :", Ratio)
print("#------------------------#")
print("##########################")

# Record Video
out = cv2.VideoWriter('Record_no_Erode.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (IMG_WIDTH, IMG_HEIGHT))

# Make Model
model = pspunet((IMG_HEIGHT, IMG_WIDTH ,3), n_classes)
model.load_weights("pspunet_weight.h5")

while True:
	ret,frame = cap.read()

	if ret == True:
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = frame[tf.newaxis, ...]   # 482 270 3
		frame = frame/255                # 1 482 270 3
		pre = model.predict(frame)       # 482 x 270 x 1
		pre = create_mask(pre).numpy()

	else:
		cv2.destroyAllWindows()
		cap.release()
		break

	# ----------------------------------------------------------------------- #
	# Draw Segmented Points
	frame2 = frame/2
	frame2[0] *= 0  # Remove Non Segmented Background
	frame2[0][(pre==6).all(axis=2)] += [1.0, 1.0, 1.0]     # "sidewalk_blocks","sidewalk_cement" , "sidewalk_soil_stone", "sidewalk_damaged","sidewalk_other"
	# ----------------------------------------------------------------------- #

	# ----------------------------------------------------------------------- #
	# Preprocess on Segmented Image
	video = np.uint8(frame2 * 255) # 1 x 360 x 640 x 3

	# Edge Detector with Canny / Line Detector with Probability Hough Line
	dst  = video[0].copy()
	gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

	# Morphology - remove small objects
	#Kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	#erosion = cv2.erode(gray, kernel, iterations = 2)
	#erosion = cv2.erode(gray, None)
	canny = cv2.Canny(gray, 50, 200, None, 3)
	lines = cv2.HoughLinesP(canny, 1, np.pi/180, 50, None, 50, 10)

	# Draw Lines on Original Image
	if lines is not None:
		for i in lines: # i [[295 118 353 118]] numpy ...
			cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 1, cv2.LINE_AA)

	# Make video
	out.write(dst)



