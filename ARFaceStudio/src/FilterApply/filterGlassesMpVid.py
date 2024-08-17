import cv2 as cv
import os
import numpy as np
import mediapipe as mp

# INITIALIZE MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# INITIALIZE MEDIAPIPE DRAWING UTILS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# LOAD FILTER IMAGE WITH TRANSPARENCY
filter_img = cv.imread(os.path.join(os.getcwd(), r'data/testImages/roundGlassesTr.png'), -1)

# CAPTURE VIDEO FROM WEBCAM
cap = cv.VideoCapture(0)

# WHILE VIDEO IS OPEN
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    try:
        # CONVERT THE FRAME TO RGB AS MEDIAPIPE EXPECTS RGB INPUT
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # PROCESS THE FRAME AND GET THE FACE LANDMARKS
        results = face_mesh.process(frame_rgb)
        
        # DRAW THE FACE LANDMARKS
        if results.multi_face_landmarks:
            
            for face_landmarks in results.multi_face_landmarks:
                
                # GET THE COORDINATES OF THE EYES
                left_eye = face_landmarks.landmark[33]  # Left eye center
                right_eye = face_landmarks.landmark[263]  # Right eye center
                
                # CONVERT LANDMARKS TO PIXEL COORDINATES
                h, w, _ = frame.shape
                left_eye_coords = int(left_eye.x * w), int(left_eye.y * h)
                right_eye_coords = int(right_eye.x * w), int(right_eye.y * h)
                
                # CALCULATE THE SIZE AND POSITION OF THE FILTER
                filter_width = int((right_eye_coords[0] - left_eye_coords[0]) * 1.8) #SCALING FACTOR 
                filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
                filter_x = int(left_eye_coords[0] - filter_width / 5) # CENTER FILTER HORIZONTALLY
                filter_y = int(left_eye_coords[1] - filter_height / 2) # ADJUSTED VERTICAL POSITIONING

                # RESIZE THE FILTER TO MATCH THE CALCULATED DIMENSIONS
                resized_filter = cv.resize(filter_img, (filter_width, filter_height), interpolation=cv.INTER_AREA)

                # EXTRACT THE REGIONS WHERE THE FILTER WILL BE PLACED
                roi = frame[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width]

                # CHECK IF THE RESIZED FILTER HAS AN ALPHA CHANNEL
                if resized_filter.shape[2] == 4:
                    # SPLIT THE FILTER INTO ITS RGB AND ALPHA CHANNELS
                    filter_rgb = resized_filter[:, :, :3]
                    filter_alpha = resized_filter[:, :, 3]

                    # CREATE A MASK AND ITS INVERSE FROM THE ALPHA CHANNEL
                    _, mask = cv.threshold(filter_alpha, 1, 255, cv.THRESH_BINARY)
                    mask_inv = cv.bitwise_not(mask)
                else:
                    # IF NO ALPHA CHANNEL, USE THE FILTER AS IS AND CREATE A MASK FROM IT
                    filter_rgb = resized_filter
                    filter_gray = cv.cvtColor(resized_filter, cv.COLOR_BGR2GRAY)
                    _, mask = cv.threshold(filter_gray, 1, 255, cv.THRESH_BINARY)
                    mask_inv = cv.bitwise_not(mask)

                # ENSURE THE MASK AND ROI SIZES ARE THE SAME
                if roi.shape[:2] != mask.shape[:2]:
                    print("SIZE MISMATCH BETWEEN ROI AND FILTER MASK")
                    continue

                # BLACK-OUT THE AREA OF THE FILTER IN THE ROI
                img_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

                # TAKE ONLY THE REGION OF THE FILTER FROM THE FILTER IMAGE
                filter_fg = cv.bitwise_and(filter_rgb, filter_rgb, mask=mask)

                # ADD THE FILTER TO THE ROI AND MODIFY THE MAIN IMAGE
                dst = cv.add(img_bg, filter_fg)
                frame[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width] = dst

        else:
            print("No face landmarks detected")

    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # DISPLAY THE FRAME
    cv.imshow('MediPipe Face Mesh', cv.flip(frame, 1))
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# RELEASE THE CAPTURE AND CLOSE ALL WINDOWS
cap.release()
cv.destroyAllWindows()