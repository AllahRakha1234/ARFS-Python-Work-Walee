import os
import cv2 as cv
import mediapipe as mp


# INITIALIZE MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# INITIALIZE MEDIAPIPE DRAWING
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# VIDEO CAPTURE
cap = cv.VideoCapture(0)


# WHILE LOOP FOR MAIN PROCESSING
while cap.isOpened():
    ret, frame = cap.read()
    
    # CONTINUE IF NO FRAME
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    # TRY-CATCH BLOCK FOR PROCESSING FRAME
    try:
        
        # PROCESSING
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_results = face_mesh.process(frame_rgb)
        
        # GETTING FACE RESULTS
        if frame_results.multi_face_landmarks:
            
            face_results = frame_results.multi_face_landmarks
            
            # DRAWING FACE MESH
            for face_landmarks in face_results:
                
                # GETTING REQUIRED LANDMARKS
                
                # left_eye_indices = [33, 133]
                # right_eye_indices = [362, 263]
                # nose_indices = [1, 2, 98, 327]
                # lips_indices = [61, 291, 78, 308]
                # left_ear_indices = [234, 93, 132]
                # right_ear_indices = [454, 323, 361]

                # # Get landmark coordinates for these features
                # left_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                #             int(face_landmarks.landmark[i].y * frame.shape[0])) for i in left_eye_indices]
                # right_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                #             int(face_landmarks.landmark[i].y * frame.shape[0])) for i in right_eye_indices]
                # nose = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                #         int(face_landmarks.landmark[i].y * frame.shape[0])) for i in nose_indices]
                # lips = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                #         int(face_landmarks.landmark[i].y * frame.shape[0])) for i in lips_indices]
                # left_ear = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                #             int(face_landmarks.landmark[i].y * frame.shape[0])) for i in left_ear_indices]
                # right_ear = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                #             int(face_landmarks.landmark[i].y * frame.shape[0])) for i in right_ear_indices]

                # # Draw circles for each landmark point on the frame
                # for point in left_eye + right_eye + nose + lips + left_ear + right_ear:
                #     cv.circle(frame, point, 2, (0, 255, 255), -1)

                # ----------------------------------------------------------
                
                
                # DRAWING
                mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
                
                
                mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                )   
            
                      
        else:
            print("No face landmarks detected")
       
    # EXCEPTIONS
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    # SHOW FRAME
    cv.imshow("Face Mesh", cv.flip(frame, 1))
        
    # BREAK  IF 'q' IS PRESSED
    if cv.waitKey(1) == ord('q'):
        break



# RELEASE CAPTURE AND DESTROY ALL WINDOWS
cap.release()
cv.destroyAllWindows()