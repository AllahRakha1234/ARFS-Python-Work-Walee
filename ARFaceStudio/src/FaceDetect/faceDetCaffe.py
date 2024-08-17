import cv2
import os

# LOAD THE PRE-TRAINED MODEL AND CONFIGURATION FILE
modelFile = os.path.join(os.getcwd(), 'models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
configFile = os.path.join(os.getcwd(), 'models/deploy.prototxt.txt')

# INITIALIZE THE DNN FACE DETECTOR
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# LOAD AN IMAGE
image = cv2.imread(os.path.join(os.getcwd(), 'data/testImages/group3.jpeg'))
(h, w) = image.shape[:2]

# PREPARE THE IMAGE FOR THE NETWORK
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# PERFORM FACE DETECTION
net.setInput(blob)
detections = net.forward()

# print(detections.shape)
# print(detections.shape[2])

# LOOP OVER THE DETECTIONS AND DRAW BOUNDING BOXES
for i in range(detections.shape[2]): # detections = (1, 1, 200, 7) | detections.shape[2] = number of maximum detections that can be made
    confidence = detections[0, 0, i, 2] # detections[0, 0, i, 2] = confidence of the ith detection
    if confidence > 0.5:  # CONFIDENCE THRESHOLD
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# DISPLAY THE OUTPUT IMAGE
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()














# MODEL PARAMETERS EXPLAINATION

"""
(*** Remeber these detection array values locations differ from model to model. That's why we need to check the model documentation to get the exact values. Here is also different. ***)
 
Breakdown of Detection Details

For each detection in the detections array, the structure is:

detections[0, 0, i, 2]: Confidence score of the detection. This value indicates how confident the model is that the detected region contains a face. Higher values represent higher confidence.

detections[0, 0, i, 1]: X-coordinate of the top-left corner of the bounding box.

detections[0, 0, i, 2]: Y-coordinate of the top-left corner of the bounding box.

detections[0, 0, i, 3]: X-coordinate of the bottom-right corner of the bounding box.

detections[0, 0, i, 4]: Y-coordinate of the bottom-right corner of the bounding box.

detections[0, 0, i, 5]: Width of the bounding box.

detections[0, 0, i, 6]: Height of the bounding box.

Example Interpretation
If detections[0, 0, 0] is:

[0.98, 0.12, 0.20, 0.40, 0.60, 0.28, 0.38]

Confidence Score: 0.98 (98% confidence that a face is detected).
Bounding Box Coordinates:
Top-left corner: (0.12, 0.20) (normalized coordinates).
Bottom-right corner: (0.40, 0.60) (normalized coordinates).
Bounding Box Dimensions:
Width: 0.28 (normalized width).
Height: 0.38 (normalized height).

-------------------------------------------------------------------------------------

Code:
box = detections[0, 0, i, 3:7] * [w, h, w, h]
(startX, startY, endX, endY) = box.astype("int")


Explanation:

detections[0, 0, i, 3:7]:
Purpose: Extracts the bounding box coordinates for the i-th detection.
Explanation: Given the shape (1, 1, 200, 7) of the detections array, detections[0, 0, i, 3:7] retrieves the bounding box coordinates for the i-th detection.
3:7 slices the array to get the values for the bounding box coordinates: [x1, y1, x2, y2] (the normalized coordinates of the top-left and bottom-right corners of the bounding box).
* [w, h, w, h]:

Purpose: Converts normalized coordinates to pixel coordinates.
Explanation: The bounding box coordinates extracted from detections are normalized, so they need to be scaled to the original image dimensions. The * [w, h, w, h] operation multiplies these normalized coordinates by the width (w) and height (h) of the original image to convert them to pixel values.
w and h are the width and height of the image.
The operation scales the x-coordinates by w and the y-coordinates by h.
box.astype("int"):

Purpose: Converts the bounding box coordinates from floating-point to integer values.
Explanation: The result of the multiplication will be in floating-point format. The astype("int") method converts these values to integers, which is necessary for drawing bounding boxes with pixel coordinates.
(startX, startY, endX, endY):

Purpose: Unpacks the pixel coordinates into individual variables.
Explanation: After converting the bounding box coordinates to integers, they are unpacked into four variables: startX, startY, endX, and endY. These variables represent:
startX: X-coordinate of the top-left corner of the bounding box.
startY: Y-coordinate of the top-left corner of the bounding box.
endX: X-coordinate of the bottom-right corner of the bounding box.
endY: Y-coordinate of the bottom-right corner of the bounding box.

"""