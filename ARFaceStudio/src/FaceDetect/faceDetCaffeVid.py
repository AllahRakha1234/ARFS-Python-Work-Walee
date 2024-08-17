import cv2
import os

# LOAD THE PRE-TRAINED MODEL AND CONFIGURATION FILE
modelFile = os.path.join(os.getcwd(), 'models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
configFile = os.path.join(os.getcwd(), 'models/deploy.prototxt.txt')

# INITIALIZE THE DNN FACE DETECTOR
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# OPEN A CONNECTION TO THE WEBCAM
cap = cv2.VideoCapture(0)

while True:
    # CAPTURE FRAME-BY-FRAME
    ret, frame = cap.read()
    
    # BREAK THE LOOP IF NO FRAME IS CAPTURED
    if not ret:
        break

    # GET FRAME DIMENSIONS
    (h, w) = frame.shape[:2]

    # PREPARE THE FRAME FOR THE NETWORK
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # PERFORM FACE DETECTION
    net.setInput(blob)
    detections = net.forward()

    # LOOP OVER THE DETECTIONS AND DRAW BOUNDING BOXES
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # CONFIDENCE THRESHOLD
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # DISPLAY THE FRAME WITH DETECTED FACES
    cv2.imshow("Face Detection", cv2.flip(frame, 1))

    # BREAK THE LOOP IF 'q' IS PRESSED
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# RELEASE THE CAPTURE AND CLOSE WINDOWS
cap.release()
cv2.destroyAllWindows()
