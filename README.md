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
Please download the source code of hand gesture recognition project: [Hand Gesture Recognition Code](https://github.com/ManojeetMahato/HAND-GESTURE-RECOGNITION-SYSTEM)

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
```

### Step 2 - Initialize Models:
## Initialize Mediapipe
```python
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
```
## Initialize TensorFlow
```python
# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
```
#### Output
['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

### Step 3 – Read frames from a webcam:
```python
# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

while True:
  # Read each frame from the webcam
  _, frame = cap.read()
x , y, c = frame.shape

  # Flip the frame vertically
  frame = cv2.flip(frame, 1)
  # Show the final output
  cv2.imshow("Output", frame)
  if cv2.waitKey(1) == ord('q'):
    		break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
```

### Step 4 – Detect hand keypoints:
```python
framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Get hand landmark prediction
  result = hands.process(framergb)

  className = ''

  # post process the result
  if result.multi_hand_landmarks:
    	landmarks = []
    	for handslms in result.multi_hand_landmarks:
        	for lm in handslms.landmark:
            	# print(id, lm)
            	lmx = int(lm.x * x)
            	lmy = int(lm.y * y)

            	landmarks.append([lmx, lmy])

        	# Drawing landmarks on frames
        	mpDraw.draw_landmarks(frame, handslms, 
mpHands.HAND_CONNECTIONS)
```

### Step 5 – Recognize hand gestures:
```pyhton
# Predict gesture in Hand Gesture Recognition project
        	prediction = model.predict([landmarks])
print(prediction)
        	classID = np.argmax(prediction)
        	className = classNames[classID]

  # show the prediction on the frame
  cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
               	1, (0,0,255), 2, cv2.LINE_AA)
```

### Results

![image](https://user-images.githubusercontent.com/75213442/233597990-577d3d0c-3662-4b06-809f-4b71be4d0cc1.png)
![image](https://user-images.githubusercontent.com/75213442/233598112-c16fa174-b2dd-4d6e-b054-e8ac313e38e1.png)

