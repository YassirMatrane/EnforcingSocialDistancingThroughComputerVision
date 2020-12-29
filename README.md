# covid-social-distancing-detection

This project is a social distancing detector implemented in Python with OpenCV and Tensorflow.
The result that can be obtained is the following :

![](/img/result.gif)

# Installation

### OpenCV
If you are working under a Linux distribution or a MacOS, use this [tutorial](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/) from Adrian Rosebrock to install this library.

### Other requirements
All the other requirements can be installed via the command : 
```bash
pip install -r requirements.txt
```

# Download Tensorflow models

In my project I used the faster_rcnn_inception_v2_coco model. I could not upload it to github because it is to heavy. You can download this model and several others from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
Just download all the models you want to try out, put them in the models folder and unzip them. For example :
```bash
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

# Run project

### Calibrate
Run 
```bash
python calibrate_with_mouse.py
```
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

# Outputs

VideoRight1
The real distance between p5 and p6 is 224 in horizontal direction
The real distance between p5 and p7 is 330 in vertical direction

videoLeft1
The real distance between p5 and p6 is 194 in horizontal direction
The real distance between p5 and p7 is 220 in vertical direction
