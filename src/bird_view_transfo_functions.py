import numpy as np
import cv2

class Preprocessing:		
	def get_human_box_detection(self,frame,boxes,scores,classes,height,width,threshold):
		array_boxes = list() # Create an empty list
		#green = (0, 255, 0)
		for i in range(boxes.shape[1]):
			# If the class of the detected object is 1 and the confidence of the prediction is > threshold
			if int(classes[i]) == 1 and scores[i] > threshold:
				# Multiply the X coordonnate by the height of the image and the Y coordonate by the width
				# To transform the box value into pixel coordonate values.
				box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
				# Add the results converted to int
				array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
		#for i in range(len(array_boxes)):
		#	frame = cv2.rectangle(frame,(array_boxes[i][1],array_boxes[i][0]),(array_boxes[i][3],array_boxes[i][2]),green,2)
		return array_boxes
	
	def get_centroids_and_groundpoints(self,array_boxes_detected):
		array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
		for index,box in enumerate(array_boxes_detected):
			# Draw the bounding box 
			# Get the both important points
			centroid,ground_point = self.get_points_from_box(box)
			array_centroids.append(centroid)
			array_groundpoints.append(ground_point)
		return array_groundpoints,array_centroids

	def get_points_from_box(self,box):
		# Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
		center_x = int(((box[1]+box[3])/2))
		center_y = int(((box[0]+box[2])/2))
		# Coordiniate on the point at the bottom center of the box
		center_y_ground = center_y + ((box[2] - box[0])/2)
		return (center_x,center_y),(center_x,int(center_y_ground))

	def compute_perspective_transform(self,corner_points,width,height,image):
		# Create an array out of the 4 corner points
		corner_points_array = np.float32(corner_points)
		# Create an array with the parameters (the dimensions) required to build the matrix
		img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
		# Compute and return the transformation matrix
		matrix = cv2.getPerspectiveTransform(corner_points_array,img_params) 
		img_transformed = cv2.warpPerspective(image,matrix,(width,height))
		return matrix,img_transformed

	def compute_point_perspective_transformation(self,matrix,list_downoids):
		# Compute the new coordinates of our points
		list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
		transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
		# Loop over the points and add them to the list that will be returned
		transformed_points_list = list()
		try:
			for i in range(0,transformed_points.shape[0]):
				transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
		except:
			print("0 pedestrians")
		return transformed_points_list

	def compute_scale_factor_perspective_transformation(self,pts,prespective_transform,wRealDis,hRealDis):
		pts = np.float32(np.array(pts))
		warped_pt = cv2.perspectiveTransform(pts[None, :, :], prespective_transform)[0]
		distance_w = int(np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2))
		distance_h = int(np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2))
		return float(int(wRealDis)/distance_w), float(int(hRealDis)/distance_h)

	def cal_dis(self,p1, p2, scaleF_w, scaleF_h):		
		h = abs(p2[1]-p1[1])
		w = abs(p2[0]-p1[0])		
		dis_w = float(w*scaleF_w)
		dis_h = float(h*scaleF_h)		
		return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))

	#Function calculates distance between all pairs and calculates closeness ratio.
	def get_distances(self,boxes1, bottom_points, distance_w, distance_h,distance_minimum):		
		distance_mat = []
		bxs = []		
		for i in range(len(bottom_points)):
			for j in range(len(bottom_points)):
				if i != j:
					dist = self.cal_dis(bottom_points[i], bottom_points[j], distance_w, distance_h)
					if dist < distance_minimum:
						closeness = 0
						distance_mat.append([bottom_points[i], bottom_points[j], closeness])
						bxs.append([boxes1[i], boxes1[j], closeness])
					elif dist >= distance_minimum and dist < distance_minimum + 20:
						closeness = 1
						distance_mat.append([bottom_points[i], bottom_points[j], closeness])
						bxs.append([boxes1[i], boxes1[j], closeness])       
					elif dist > distance_minimum:
						closeness = 2
						distance_mat.append([bottom_points[i], bottom_points[j], closeness])
						bxs.append([boxes1[i], boxes1[j], closeness])					
		return distance_mat, bxs

	def get_count(self,distances_mat):
		r = []
		g = []
		y = []
		for i in range(len(distances_mat)):
			if distances_mat[i][2] == 0:
				if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
					r.append(distances_mat[i][0])
				if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
					r.append(distances_mat[i][1])
					
		for i in range(len(distances_mat)):

			if distances_mat[i][2] == 1:
				if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
					y.append(distances_mat[i][0])
				if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
					y.append(distances_mat[i][1])
			
		for i in range(len(distances_mat)):
		
			if distances_mat[i][2] == 2:
				if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
					g.append(distances_mat[i][0])
				if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
					g.append(distances_mat[i][1])
	
		return (len(r),len(y),len(g))
