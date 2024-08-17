import cv2 as cv
import os

# LOAD PRE-TRAINED HAAR CASCADE FACE DETECTION MODEL
print(os.getcwd())
cascade_path = os.path.join(os.getcwd(), 'models/haarcascade_frontalface_default.xml')
clf = cv.CascadeClassifier(cascade_path)

# CAPTURE VIDEO FROM WEBCAM
img = cv.imread(os.path.join(os.getcwd(), r'data/testImages/group3.jpeg'))
# camera_ouptut = cv.VideoCapture(0)


while True:
    
    # CODE FOR IMAGE UPLAODED
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = clf.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, width, height) in faces:
        cv.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 2) # BGR
        
    cv.imshow('Detected Faces', img)
    
    
    # CODE FOR VIDEO CAPTURING
    
    # _, frame = camera_ouptut.read()
    
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # faces = clf.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # for (x, y, width, height) in faces:
    #     cv.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2) # BGR
        
    # cv.imshow('Detected Faces', cv.flip(frame, 1))
    
    # REMAIN SAME FOR BOTH  
    
    if cv.waitKey(1) == ord('q'):
        break
    
# camera_ouptut.release()
cv.destroyAllWindows()