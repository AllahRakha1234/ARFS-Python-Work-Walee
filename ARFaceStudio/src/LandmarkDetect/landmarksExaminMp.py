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

print("mp_drawing.FACEMESH_LEFT_EYE: ", mp_face_mesh.FACEMESH_LEFT_EYE)

# LOAD IMAGE
image_path = os.path.join(os.getcwd(), 'data/testImages/face.jpg')
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")
    
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to load image from {image_path}")

# RESIZE IMAGE TO MAXIMUM DIMENSIONS FOR BETTER VISIBILITY
max_width = 1200  # Adjust the maximum width as needed
max_height = 1200  # Adjust the maximum height as needed
h, w, _ = image.shape
scale = min(max_width / w, max_height / h)
new_w = int(w * scale)
new_h = int(h * scale)
resized_image = cv2.resize(image, (new_w, new_h))

# CONVERT RESIZED IMAGE TO RGB AS MEDIAPIPE EXPECTS RGB INPUT
resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# PROCESS THE IMAGE AND GET THE FACE LANDMARKS
results = face_mesh.process(resized_image_rgb)

# Define landmark indices and colors for specific features
landmark_indices = {
    # Eyes (left and right)
    'left_eye': (range(33, 133), (0, 255, 255)),  # Cyan
    'right_eye': (range(362, 463), (255, 0, 255)), # Magenta
    # Nose
    'nose': (range(1, 27), (0, 255, 0)),  # Green
    # Lips
    'lips': (range(48, 68), (255, 0, 0)),  # Red
    # Ears (left and right)
    'left_ear': (range(234, 241), (255, 165, 0)),  # Orange
    'right_ear': (range(454, 461), (128, 0, 128)), # Purple
}

# DRAW LANDMARK INDICES FOR SPECIFIC FEATURES
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for feature, (indices, dot_color) in landmark_indices.items():
            line_color = dot_color  # Use the same color for lines and text         
            
            for index in indices:
                landmark = face_landmarks.landmark[index]
                # Convert normalized landmark coordinates to pixel coordinates
                x, y = int(landmark.x * new_w), int(landmark.y * new_h)

                # Draw a circle on each landmark with the specified color
                cv2.circle(resized_image, (x, y), 4, dot_color, -1)  # Increased circle size for better visibility

                # Calculate dynamic line endpoint based on the image size
                offset_x = (x - new_w / 2) * 2  # Move away from the center of the face
                offset_y = (y - new_h / 2) * 2  # Move away from the center of the face
                line_end_x = int(x + offset_x)
                line_end_y = int(y + offset_y)

                # Ensure the line endpoints are within the image bounds
                line_end_x = max(0, min(line_end_x, new_w - 1))
                line_end_y = max(0, min(line_end_y, new_h - 1))

                # Draw the line with the specified color
                cv2.line(resized_image, (x, y), (line_end_x, line_end_y), line_color, 2)  # Increased line thickness

                # Draw the index number with the specified color
                cv2.putText(resized_image, str(index), (line_end_x + 10, line_end_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 1, cv2.LINE_AA)
else:
    print("No face landmarks detected")

# DISPLAY THE OUTPUT IMAGE
cv2.imshow('Face Landmarks', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
