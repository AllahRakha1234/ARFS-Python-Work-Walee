import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Use static image mode for processing a single image
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils

# Print available landmark constants
all_landmark_indices = {
    # 'mp_face_mesh.FACEMESH_CONTOURS': mp_face_mesh.FACEMESH_CONTOURS,
    # 'mp_face_mesh.FACEMESH_TESSELATION': mp_face_mesh.FACEMESH_TESSELATION,
    # 'mp_face_mesh.FACEMESH_VERTEXTES': mp_face_mesh.FACEMESH_VERTEXTES,
    # Eyes (left and right)
    'mp_face_mesh.FACEMESH_LEFT_EYE': mp_face_mesh.FACEMESH_LEFT_EYE,
    'mp_face_mesh.FACEMESH_RIGHT_EYE': mp_face_mesh.FACEMESH_RIGHT_EYE,
    # Eyebrows (left and right)
    'mp_face_mesh.FACEMESH_LEFT_EYEBROW': mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    'mp_face_mesh.FACEMESH_RIGHT_EYEBROW': mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    # Nose
    'mp_face_mesh.FACEMESH_NOSE': mp_face_mesh.FACEMESH_NOSE,
    # Lips (upper and lower)
    'mp_face_mesh.FACEMESH_LIPS': mp_face_mesh.FACEMESH_LIPS,
    # 'mp_face_mesh.FACEMESH_UPPER_LIP': mp_face_mesh.FACEMESH_UPPER_LIP,
    # 'mp_face_mesh.FACEMESH_LOWER_LIP': mp_face_mesh.FACEMESH_LOWER_LIP,
    # Teeth (upper and lower)
    # 'mp_face_mesh.FACEMESH_TEETH': mp_face_mesh.FACEMESH_TEETH,
    # Tongue
    # 'mp_face_mesh.FACEMESH_TONGUE': mp_face_mesh.FACEMESH_TONGUE,
    # Face contour
    'mp_face_mesh.FACEMESH_FACE_OVAL': mp_face_mesh.FACEMESH_FACE_OVAL,
    # Ears (left and right)
    # 'mp_face_mesh.FACEMESH_LEFT_EAR': mp_face_mesh.FACEMESH_LEFT_EAR,
    # 'mp_face_mesh.FACEMESH_RIGHT_EAR': mp_face_mesh.FACEMESH_RIGHT_EAR,
   
}

for name, landmark_indices in all_landmark_indices.items():
    if landmark_indices is not None:
        print(f"{name}: {landmark_indices}")
