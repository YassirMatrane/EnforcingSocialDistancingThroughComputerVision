import cv2
import numpy as np
import yaml
import imutils

 
# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    global img
    global index
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(list_points) < 4:
            cv2.circle(img,(x,y),5,(0,255,0),-1)
            if len(list_points) >= 1 and len(list_points) <= 3:
                cv2.line(img, (x, y), (list_points[len(list_points)-1][0], list_points[len(list_points)-1][1]), (70, 70, 70), 2)
                if len(list_points) == 3:
                    cv2.line(img, (x, y), (list_points[0][0], list_points[0][1]), (70, 70, 70), 2)
            list_points.append([x,y])
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
            label = "p" + str(index)
            cv2.putText(img,label,(x+4,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1,cv2.LINE_AA)
            index = index + 1
            scale_factor.append([x,y])
        

##video_name = input("Enter the exact name of the video (including .mp4 or else) : ")
video_name = "PETS2009.avi"
video1 = "videoRight1.mp4"

##size_frame = input("Prompt the size of the image you want to get : ")

size_frame = 700
vs = cv2.VideoCapture("../video/"+video1)
##vs = cv2.VideoCapture(0)
# Loop until the end of the video stream
while True:    
    # Load the frame and test if it has reache the end of the video
    (frame_exists, frame) = vs.read()
    frame = imutils.resize(frame, width=int(size_frame+400),height=700)
    cv2.imwrite("../img/static_frame_from_video.jpg",frame)
    break

# Create a black image and a window
windowName = 'MouseCallback'
cv2.namedWindow(windowName)


# Load the image 
img_path = "../img/static_frame_from_video.jpg"
img = cv2.imread(img_path)

# Get the size of the image for the calibration
width,height,_ = img.shape

# Create an empty list of points for the coordinates
list_points = list()
# Create an empty list of values for scale factor
scale_factor = list()
# Index for the  three points that will estimate scale factor
index = 5
# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)
print('Top left, Top right, Bottom Left, Bottom Right')
print("Then draw 3 points in horizontal and vertical direction of known distances in real life in cm")

if __name__ == "__main__":
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        if len(list_points) == 4: 
            #Next 3 points will define N cm(unit length) distance in horizontal and vertical direction 
            #and those should form parallel lines with ROI. In above image we can se point 5 and point 6 defines 'wRealDistance' cm 
            #in real life in horizontal direction and point 5 and point 7 defines 'wRealDistance' cm in real life in vertical direction.
            if len(scale_factor) == 4:
                wRealDistance = input("Enter the actual distance between p5 and p6: ")
                hRealDistance = input("Enter the actual distance between p6 and p7: ")
            # Return a dict to the YAML file
                # Scale_factor_w between p5 and p6
                # Scale_factor_h between p5 and p7
                scale_factor.pop()
                config_data = dict(
                    image_parameters = dict(
                        p2 = list_points[3],
                        p1 = list_points[2],
                        p4 = list_points[0],
                        p3 = list_points[1],
                        scale_factor_points = scale_factor,
                        wRealDistance = wRealDistance,
                        hRealDistance = hRealDistance,
                        width_og = width,
                        height_og = height,
                        img_path = img_path,
                        size_frame = size_frame,
                        ))
                # Write the result to the config file
                with open('../conf/config_birdview.yml', 'w') as outfile:
                    yaml.dump(config_data, outfile, default_flow_style=False)
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()