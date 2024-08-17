import cv2 as cv
import dlib
import os

# Load image
image_path = os.path.join(os.getcwd(), r'data/testImages/group1.jpeg')
image = cv.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit(1)

# Convert the image to RGB format
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# Detect faces in the RGB image
faces = detector(rgb_image)

# Draw rectangles around detected faces
for face in faces:
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2) # BGR

# Show the output image
cv.imshow('Detected Faces', image)
cv.waitKey(0)
cv.destroyAllWindows()
