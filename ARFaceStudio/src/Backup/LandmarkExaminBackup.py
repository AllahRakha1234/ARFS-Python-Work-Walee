import os
import cv2
import mediapipe as mp

# INITIALIZE MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Use static image mode for processing a single image
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# INITIALIZE MEDIAPIPE DRAWING UTILS
mp_drawing = mp.solutions.drawing_utils

# LOAD IMAGE
image_path = os.path.join(os.getcwd(), 'data/testImages/face.jpg')
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")
    
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to load image from {image_path}")

# CONVERT IMAGE TO RGB AS MEDIAPIPE EXPECTS RGB INPUT
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# PROCESS THE IMAGE AND GET THE FACE LANDMARKS
results = face_mesh.process(image_rgb)

# DRAW LANDMARK INDICES
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for index, landmark in enumerate(face_landmarks.landmark):
            # Convert normalized landmark coordinates to pixel coordinates
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)

            # Draw a circle on each landmark for better visibility
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Calculate dynamic line endpoint based on the image size
            offset_x = (x - w / 2) * 2  # Move away from the center of the face
            offset_y = (y - h / 2) * 2  # Move away from the center of the face
            line_end_x = int(x + offset_x)
            line_end_y = int(y + offset_y)

            # Ensure the line endpoints are within the image bounds
            line_end_x = max(0, min(line_end_x, w - 1))
            line_end_y = max(0, min(line_end_y, h - 1))

            # Draw the line and the index number
            cv2.line(image, (x, y), (line_end_x, line_end_y), (0, 255, 0), 1)
            cv2.putText(image, str(index), (line_end_x + 5, line_end_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

else:
    print("No face landmarks detected")

# DISPLAY THE OUTPUT IMAGE
cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
