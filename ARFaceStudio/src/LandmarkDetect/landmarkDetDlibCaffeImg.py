import cv2 as cv
import dlib
import os

# LOAD THE FACE DETECTION MODEL
modelFile = os.path.join(os.getcwd(), 'models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
configFile = os.path.join(os.getcwd(), 'models/deploy.prototxt.txt')
net = cv.dnn.readNetFromCaffe(configFile, modelFile)

# LOAD THE FACIAL LANDMARK DETECTOR MODEL
landmark_detector_path = os.path.join(os.getcwd(), 'models/shape_predictor_68_face_landmarks.dat')
dlib_landmark_detector = dlib.shape_predictor(landmark_detector_path)

# UPLOADING IMAGE
img = cv.imread(os.path.join(os.getcwd(), r'data/testImages/group3.jpeg'))

while True:
    # CONVERT THE FRAME TO A BLOB
    blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # PERFORM FACE DETECTION
    net.setInput(blob)
    detections = net.forward()
    
    h, w = img.shape[:2]
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:  # CONFIDENCE THRESHOLD
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            
            # USE DLIB TO DETECT LANDMARKS ON THE DETECTED FACE
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            face_rect = dlib.rectangle(startX, startY, endX, endY)
            face_landmarks = dlib_landmark_detector(gray, face_rect)
            
            for landmark_no in range(0, 68):
                x = face_landmarks.part(landmark_no).x
                y = face_landmarks.part(landmark_no).y
                cv.circle(img, (x, y), 1, (255, 0, 0), 1)
    
    cv.imshow('Face and Landmark Detection', img)
    
    if cv.waitKey(1) == ord('q'):
        break

# RELEASE CAPTURE AND CLOSE WINDOWS
cv.destroyAllWindows()
