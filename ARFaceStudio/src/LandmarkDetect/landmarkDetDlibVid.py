# IMPORTING LIBRARIES
import cv2 as cv
import dlib
import os



# LOADING MODEL
hog_face_detector = dlib.get_frontal_face_detector()
landmark_detector_path = os.path.join(os.getcwd(), 'models/shape_predictor_68_face_landmarks.dat')
dlib_landmark_detector = dlib.shape_predictor(landmark_detector_path)

# CAPTURING VIDEO FROM WEBCAM
cap = cv.VideoCapture(0)

# LOOP TO CAPTURE FRAMES
while True:
    
    _, frame = cap.read()
    
    # FLIP THE FRAME HORIZONTALLY
    frame = cv.flip(frame, 1)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    
    for face in faces:
        
        face_landmarks = dlib_landmark_detector(gray, face)
        
        for landmark_no in range(0,68):
            x = face_landmarks.part(landmark_no).x
            y = face_landmarks.part(landmark_no).y
            cv.circle(frame, (x, y), 1, (255, 0, 0), 2)
            
    cv.imshow('Landmark Detection', frame)
    if(cv.waitKey(1) == ord('q')):
        break
    
cap.release()
cv.destroyAllWindows()
    