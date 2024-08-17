# IMPORTING LIBRARIES
import cv2 as cv
import dlib
import os



# LOADING MODEL
hog_face_detector = dlib.get_frontal_face_detector()
landmark_detector_path = os.path.join(os.getcwd(), 'models/shape_predictor_68_face_landmarks.dat')
dlib_landmark_detector = dlib.shape_predictor(landmark_detector_path)

# UPLOADING IMAGE
img = cv.imread(os.path.join(os.getcwd(), r'data/testImages/group1.jpeg'))


# LOOP TO CAPTURE FRAMES
while True:
    
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    
    for face in faces:
        
        face_landmarks = dlib_landmark_detector(gray, face)
        
        for landmark_no in range(0,68):
            x = face_landmarks.part(landmark_no).x
            y = face_landmarks.part(landmark_no).y
            cv.circle(img, (x, y), 1, (255, 0, 0), 1)
            
    cv.imshow('Landmark Detection', img)
    if(cv.waitKey(1) == ord('q')):
        break
    
cv.destroyAllWindows()
    