# HAND-GESTURE-RECOGNITION-SYSTEM
## Prerequisites for this project:
1. Python – 3.x (we used Python 3.8.8 in this project)
2. OpenCV – 4.5
   * Run “pip install opencv-python” to install OpenCV.
3. MediaPipe – 0.8.5
   * Run “pip install mediapipe” to install MediaPipe.
4. Tensorflow – 2.5.0
   * Run “pip install tensorflow” to install the tensorflow module.
5. Numpy – 1.19.3

## Download Hand Gesture Recognition Project Code
Please download the source code of hand gesture recognition project: [Hand Gesture Recognition ML Project Code](https://github.com/ManojeetMahato/HAND-GESTURE-RECOGNITION-SYSTEM)

## Steps to solve the project:
1. Import necessary packages.
2. Initialize models.
3. Read frames from a webcam.
4. Detect hand keypoints.
5. Recognize hand gestures.

### Step 1 – Import necessary packages:
```python
# import necessary packages for hand gesture recognition project using Python OpenCV
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

### Step 2 - Initialize Models:
```python
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
