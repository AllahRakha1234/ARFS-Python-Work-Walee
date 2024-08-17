import cv2
import os
import mediapipe as mp

# INITIALIZE MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh( # 468 landmarks provided by the model
    static_image_mode=True,
    max_num_faces=1,  
    # max_num_faces=10,  
    refine_landmarks=True,
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5
    # min_detection_confidence=0.1,  
    # min_tracking_confidence=0.1
)

# INITIALIZE MEDIAPIPE DRAWING UTILS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# UPLOADING IMAGE
img = cv2.imread(os.path.join(os.getcwd(), r'data/testImages/manTilted1.jpg'))
# img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)

if img is None:
    print("Failed to upload image")
else:
    try:
        # CONVERT THE FRAME TO RGB AS MEDIAPIPE EXPECTS RGB INPUT
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # PROCESS THE FRAME AND GET THE FACE LANDMARKS
        results = face_mesh.process(frame_rgb)
        
        # DRAW THE FACE LANDMARKS
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    # landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),     # OR
                    landmark_drawing_spec=None, 
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None, 
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
                
                mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                # connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )
                
            print("Face landmarks detected")
        else:
            print("No face landmarks detected")

        # DISPLAY THE FRAME
        cv2.imshow('MediPipe Face Mesh', img)
        cv2.waitKey(0)  # Wait until a key is pressed

    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

cv2.destroyAllWindows()
