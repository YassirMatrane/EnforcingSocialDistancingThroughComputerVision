# covid-social-distancing-detection

This project is a social distancing detector implemented in Python with OpenCV and Tensorflow.


# Installation

### OpenCV
If you are working under a Linux distribution or a MacOS, use this [tutorial](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/) from Adrian Rosebrock to install this library.

### Other requirements
All the other requirements can be installed via the command : 
```bash
pip install -r requirements.txt
```

# Run project

### Calibrate
Run 
```bash
python calibrate_with_mouse.py
```
From this file, we apply our manual calibration to draw ROI and distance scale from the first frame by the “setMouseCallback” function of OpenCV.

You will be asked as input the name of the video and the size of the frame you want to work with. You must use the actual size of your frame !

You will be also asked as inputs :
- The distance in centimers between 2 persons in horizontal and vertical direction

Note : It is important to start with the top right corner, than the bottom right, then bottom left, than end by top left corner !

You can add any video to the video folder and work with that.


### Start social distancing detection
Run 
```bash
python social_distanciation_video_detection.py
```
