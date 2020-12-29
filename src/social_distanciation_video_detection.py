from bird_view_transfo_functions import Preprocessing
from plot import Plot
from tf_model_object_detection import Model 
import numpy as np
import itertools
import imutils
import time
import math
import glob
import yaml
import cv2
import os

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3
threshold = 0.30
timeSpan = 10
preprocessing = Preprocessing()
plot = Plot()
riskCounts = []

def change_color_on_topview(pair):
	"""
	Draw red circles for the designated pair of points 
	"""
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), BIG_CIRCLE, COLOR_RED, 2)
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), BIG_CIRCLE, COLOR_RED, 2)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)

def draw_rectangle(corner_points):
	# Draw rectangle box over the delimitation area
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)


######################################### 
# Load the config for the top-down view #
#########################################
with open("../conf/config_birdview.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []
for section in cfg:
	corner_points.append(cfg["image_parameters"]["p1"])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p3"])
	corner_points.append(cfg["image_parameters"]["p4"])	
	scale_factor_points = cfg["image_parameters"]["scale_factor_points"]
	wRealDistance = cfg["image_parameters"]["wRealDistance"]
	hRealDistance = cfg["image_parameters"]["hRealDistance"]
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	img_path = cfg["image_parameters"]["img_path"]
	size_frame = cfg["image_parameters"]["size_frame"]


######################################### 
#		     Select the model 			#
#########################################
#model_num = input(" Please select the number related to the model that you want : ")
#if model_num == "":
model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 
#else :
#	model_path = "../models/"+model_names_list[int(model_num)]+"/frozen_inference_graph.pb"
model = Model(model_path)


######################################### 
#		     Select the video 			#
#########################################
#video_num = input("Enter the exact name of the video (including .mp4 or else) : ")
#if video_num == "":
video_path1="../video/PETS2009.avi"  
video_path = "../video/videoRight1.mp4"
#else :
#	video_path = "../video/"+video_names_list[int(video_num)]


######################################### 
#		    Minimal distance			#
#########################################
#distance is measured in terms of cm, since we have measured the 2 distances during calibration in cm
#distance_minimum = input("Prompt the size of the minimal distance between 2 pedestrians : ")
#if distance_minimum == "":
# 6 feet = 182 cm
distance_minimum = 182


######################################### 
#     Compute transformation matrix		#
#########################################
# Compute  transformation matrix from the original frame
imgP = cv2.imread(img_path)
matrix,imgOutput = preprocessing.compute_perspective_transform(corner_points,width_og,height_og,imgP)

# Show Bird's eye view of ROI
#cv2.imshow("birdEyeView",imgOutput)
#key = cv2.waitKey(1) & 0xFF
#if key == ord("q"):
#	cv2.destroyAllWindows() 

height,width,_ = imgOutput.shape
blank_image = np.zeros((height,width,3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1] 
dim = (width, height)

######################################### 
#     Estimate scale factor	   #
#########################################

scale_factor_w,scale_factor_h = preprocessing.compute_scale_factor_perspective_transformation(scale_factor_points,matrix,wRealDistance,hRealDistance)

######################################################
#########									 #########
# 				START THE VIDEO STREAM               #
#########									 #########
######################################################
vs = cv2.VideoCapture(video_path)
fps = vs.get(cv2.CAP_PROP_FPS)


totalNoFrames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
durationInSeconds = float(totalNoFrames) / float(fps)
# Separate the video into time spans to get how many frames we should processe during the time span
framesPerTimeSpan = (timeSpan*totalNoFrames)/durationInSeconds
indexFrame = 0
j = 0
c = 0
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = None
# Loop until the end of the video stream
while True:		
	# Load the frame
	(frame_exists, frame) = vs.read()

	# Test if it has reached the end of the video
	if not frame_exists: 
		break
	else:
		# Resize the image to the correct size
		frame = imutils.resize(frame, width=int(size_frame+400),height=700)
				
		# Make the predictions for this frame
		(boxes, scores, classes) =  model.predict(frame)
		# Get the human detected in the frame and return the 2 points to build the bounding box  
		array_boxes_detected = preprocessing.get_human_box_detection(frame,boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1],threshold)
		# Both of our lists that will contain the centroïds coordonates and the ground points

		array_centroids,array_groundpoints = preprocessing.get_centroids_and_groundpoints(array_boxes_detected)

		# Use the transform matrix to get the transformed coordonates
		transformed_downoids = preprocessing.compute_point_perspective_transformation(matrix,array_groundpoints)		
		

		########
		distances_mat, bxs_mat = preprocessing.get_distances(array_boxes_detected, transformed_downoids, scale_factor_w, scale_factor_h,distance_minimum)
		risk_count = preprocessing.get_count(distances_mat)
			
		riskCounts.append(risk_count)

		# once the count of high-risk class within a timespan ( time span has framesPerTimestamp) exceeds the others, an alert state is triggered.
		if indexFrame != 0 and indexFrame%int(framesPerTimeSpan) == 0:		
			hRisk = 0
			lRisk = 0
			noRisk = 0	
			for i in range(j,int(indexFrame)):
				if i > len(risk_count):
					break
				hRisk = hRisk + riskCounts[i][0]
				lRisk = hRisk + riskCounts[i][1]
				noRisk = hRisk + riskCounts[i][2]
			j = j + int(framesPerTimeSpan)
			if hRisk > noRisk:
				print("Be vigilant, you are breaking the social distancing at the timeSpan ",j/framesPerTimeSpan)
		indexFrame = indexFrame + 1


		

		#######
		frameResult = plot.social_distancing_view(frame, bxs_mat, array_boxes_detected, risk_count)
		c = c  + 1
		img = "frame" + str(c)
		cv2.imwrite("../images/"+img+".jpg",frameResult)	

		########
	# Draw the green rectangle to delimitate the detection zone
	#draw_rectangle(corner_points)
	# Show both images	
	cv2.imshow("Bird view", frameResult)
	##cv2.imshow("Original picture", frame)

	key = cv2.waitKey(1) & 0xFF	

	if out is None :
		out = cv2.VideoWriter('../output/videoOutput.avi',fourcc, 20.0, (frameResult.shape[1], frameResult.shape[0]),True)
	if out is not None:
		out.write(frameResult)
	# Break the loop
	if key == ord("q"):
		break


vs.release()
out.release()
cv2.destroyAllWindows()
#### statistics count ###
plot.plotStatistic(riskCounts,framesPerTimeSpan,durationInSeconds,timeSpan)

