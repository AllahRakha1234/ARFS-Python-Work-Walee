# IMPORTING LIBRARIES
import cv2 as cv
import os

# READING IMAGE
path = os.path.join(os.getcwd(), r'data/testImages/img2.jpeg')  
img = cv.imread(path)

if img is not None:
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Error: Image not found or unable to read the image.")
