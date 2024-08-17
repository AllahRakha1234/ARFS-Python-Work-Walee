import os
import cv2 as cv
import itertools
import numpy as np
import mediapipe as mp

# INITIALIZING MP MODEL
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

# GET FACE LANDMARKS FUNCTION
def getFaceLandmarks(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        return results
    else:
        return []
    
# DRAW FACE LANDMARKS FUNCTION
def drawFaceLandmarks(img, face_landmarks):
    # DRAWING
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
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
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
    )
    
    # SHOW THE IMAGE
    cv.imshow("Face Landmarks", cv.flip(img, 1))
    
    
# GETTING SIZE FUNCITON (FACE PART)
def getFacePartSize(img, face_landmarks, face_part_indices):
    
    # RETRIEVE THE HEIGHT, WIDTH, AND CHANNELS OF THE IMAGE
    height, width, _ = img.shape
    
    # CONVERTING INDICES TO A FLAT LIST
    face_part_indices_list = list(itertools.chain(*face_part_indices))
    
    # ARRAY TO STORE THE LANDMARKS OF FACE PART
    face_part_landmarks = []
    
    # GETTING THE COORDINATES OF THE FACE PART
    for idx in face_part_indices_list:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * width), int(landmark.y * height) # landmark.x = normalized values
        face_part_landmarks.append((x, y)) # Here x = actual pixel position of the landmark in the image
        
    # GETTING THE BOUNDING BOX OF THE FACE PART TO GET WIDTH AND HEIGHT
    x, y, width, height = cv.boundingRect(np.array(face_part_landmarks))
    
    # CONVERTING FACE PART LANDMARKS TO NUMPY ARRAY
    face_part_landmarks = np.array(face_part_landmarks)
    
    # RETURNING THE FACE PART LANDMARKS, WIDTH, AND HEIGHT
    return face_part_landmarks, width, height
    
# GETTING FACE PART STATE (OPEN OR CLOSE)
def isOpen(img, face_mesh_results, face_part_name, threshold=5):
    
    # GETTING HEIGHT, WIDTH, AND CHANNELS OF THE IMAGE
    height, width, _ = img.shape
    
    # CHECKING IF FACE PART NAME IS VALID
    if(face_part_name == 'MOUTH'):
        face_part_indices = mp_face_mesh.FACEMESH_LIPS
    elif(face_part_name == 'LEFT_EYE'):
        face_part_indices = mp_face_mesh.FACEMESH_LEFT_EYE
    elif(face_part_name == 'RIGHT_EYE'):
        face_part_indices = mp_face_mesh.FACEMESH_RIGHT_EYE
    else:
        return 
    
    # ITERATING THROUGH THE FOUND FACES
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        
        # GETTING THE HEIGHT OF WHOLE FACE
        _, _, face_height = getFacePartSize(img, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)
        
        # GETTING THE HEIGHT OF FACE PART
        _, _, face_part_height = getFacePartSize(img, face_landmarks, face_part_indices)
        
        # CALCULATING RATIO OF FACE PART HEIGHT TO FACE HEIGHT TO SEE IF FACE PART IS OPEN OR CLOSE
        if((face_part_height / face_height) * 100 > threshold):
            return "OPEN"
        else:
            pass
            # print(f"{face_part_name} is close")
    

# OVERLAY FUNCTION
def overlay(img, filter_img, face_landmarks, face_part_name, face_part_indices):
    
    # CREATING COPY OF INPUT IMAGE
    annotated_img = img.copy()
    
    # TRY-CATCH BLOCK TO AVOID ERROR WHEN RESIZING IMAGES
    try:
        # FILTER IMAGE HEIGHT AND WIDTH
        filter_img_height, filter_img_width, _ = filter_img.shape
        
        # GETTING HEIGHT OF THE FACE PART ON WHICH WE WILL OVERLAY THE FILTER IMAGE
        face_part_landmarks, face_part_height, _ = getFacePartSize(img, face_landmarks, face_part_indices)
        
        # SPECIFYING HEIGHT TO WHICH THE FILTER IMAGE IS REQUIRED TO BE RESIZED
        required_filter_height = int(face_part_height * 1.2)
        
        # Resize the filter image to the required height, while keeping the aspect ratio constant
        resized_filter_img = cv.resize(filter_img, 
                                       (int(filter_img_width * (required_filter_height / filter_img_height)), 
                                        required_filter_height))
        
        # GET THE NEW WIDTH AND HEIGHT OF FILTER IMAGE
        filter_img_height, filter_img_width, _  = resized_filter_img.shape
        
        # CONVERTING IMAGE TO GRAYSCALE AND APPLY THE THRESHOLD TO GET THE MASK IMAGE
        grayscale_img = cv.cvtColor(resized_filter_img, cv.COLOR_BGR2GRAY)
        _, filter_img_mask = cv.threshold(grayscale_img, 25, 255, cv.THRESH_BINARY_INV)
        
        # Ensure mask is a single channel image
        if filter_img_mask.ndim != 2:
            raise ValueError("Mask must be a single-channel image")
        
        # CALCULATING THE CENTER OF THE FACE PART
        face_part_center = face_part_landmarks.mean(axis=0).astype("int")
        
        # CHECKING IF THE FACE PART IS MOUTH
        if face_part_name == 'MOUTH':
            # CALCULATING THE LOCATION WHERE THE SMOKE FILTER WILL BE PLACED
            location = (int(face_part_center[0] - filter_img_width / 3), int(face_part_center[1]))
        else:
            # CALCULATE THE LOCATION WHERE THE EYE FILTER IMAGE WILL BE PLACED
            location = (int(face_part_center[0] - filter_img_width / 2), int(face_part_center[1] - filter_img_height / 2))
        
        # RETRIEVING THE REGION OF INTEREST FROM THE IMAGE WHERE THE FILTER IMAGE WILL BE PLACED
        ROI = img[location[1]: location[1] + filter_img_height, location[0]: location[0] + filter_img_width]
        
        # Ensure ROI and mask are the same size
        if ROI.shape[:2] != filter_img_mask.shape:
            raise ValueError("Mask size must match the region of interest (ROI) size")
        
        # PERFORMING BITWISE-AND OPERATION
        resultant_image = cv.bitwise_and(ROI, ROI, mask=filter_img_mask)
        
        # ADD THE RESULTANT IMAGE AND THE RESIZED FILTER IMAGE
        resultant_image = cv.add(resultant_image, resized_filter_img)
        
        # UPDATE THE IMAGE'S REGION OF INTEREST WITH RESULTANT IMAGE
        annotated_img[location[1]: location[1] + filter_img_height, location[0]: location[0] + filter_img_width] = resultant_image
        
    except Exception as e:
        print(f"Error occurs: {e}")
    
    return annotated_img

# DRAW REQUIRED FACE LANDMARKS FUNCTION
def drawRequiredFaceLandmarks(img, face_landmarks, landmark_indices):
    # DRAWING
    for idx in landmark_indices:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
        cv.circle(img, (x, y), 2, (0, 255, 0), -1)
    
    # SHOW THE IMAGE
    cv.imshow("Face Landmarks", cv.flip(img, 1))

# GETTING REQUIRED LANDMARKS
def getReqLandmarksIndices():
    # LANDMARKS INDICES
    left_eye_indices = list(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE))
    right_eye_indices = list(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE))
    lips_indices = list(itertools.chain(*mp_face_mesh.FACEMESH_LIPS))
    # oval_indices = list(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL))
    
    return [left_eye_indices, right_eye_indices, lips_indices]



# MAIN FUNCTION
if __name__ == "__main__":
    
    # READ IMAGES
    left_eye = cv.imread(os.path.join(os.getcwd(), r'data/eyesMouthProject/left_eye.png'))
    right_eye = cv.imread(os.path.join(os.getcwd(), r'data/eyesMouthProject/right_eye.png'))
    # READ ANIMAITON VIDEO
    smoke_animation = cv.VideoCapture(os.path.join(os.getcwd(), r'data/eyesMouthProject/smoke_animation.mp4'))
    
    # SET THE SMOKE ANIMATION VIDEO FRAME COUNTER TO ZERO.
    smoke_frame_counter = 0
    
    # CAMERA CAPTURE
    cap = cv.VideoCapture(0)
    
    # WHILE LOOP FOR MAIN PROCESSING
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BREAK THE LOOP IF NO FRAME
        if not ret:
            print("Failed to capture image")
            continue
        # SMOKE ANIMATION VIDEO LOOP CODE
        # Read a frame from smoke animation video
        _, smoke_frame = smoke_animation.read()
        # Increment the smoke animation video frame counter.
        smoke_frame_counter += 1
        # Check if the current frame is the last frame of the smoke animation video.
        if smoke_frame_counter == smoke_animation.get(cv.CAP_PROP_FRAME_COUNT):     
            # Set the current frame position to first frame to restart the video.
            smoke_animation.set(cv.CAP_PROP_POS_FRAMES, 0)
            # Set the smoke animation video frame counter to zero.
            smoke_frame_counter = 0
        
        # GET FACE LANDMARKS FUNCTION CALLING   
        face_mesh_results = getFaceLandmarks(frame)
        
        if face_mesh_results:
            
            # # GET REQUIRED LANDMARKS INDICES
            # required_landmarks_indices = getReqLandmarksIndices()
            
            # # PROCESSING EACH FACE LANDMARKS
            # for face_landmarks in face_mesh_results:
            #     face_part_landmarks, width, height = getFacePartSize(frame, face_landmarks, mp_face_mesh.FACEMESH_LEFT_EYE)
            #     print(f"Face Part Landmarks: {face_part_landmarks}")
            #     print(f"Face Part Width: {width}")
            #     print(f"Face Part Height: {height}")
            #     for landmark_indices in required_landmarks_indices:
            #         drawRequiredFaceLandmarks(frame, face_landmarks, landmark_indices)
            
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Get the mouth isOpen status of the person in the frame.
                mouth_status = isOpen(frame, face_mesh_results, 'MOUTH', 
                                            threshold=15)
                
                # Get the left eye isOpen status of the person in the frame.
                left_eye_status = isOpen(frame, face_mesh_results, 'LEFT_EYE', 
                                                threshold=4.5 )
                
                # Get the right eye isOpen status of the person in the frame.
                right_eye_status = isOpen(frame, face_mesh_results, 'RIGHT_EYE', 
                                                threshold=4.5)
                
                if(mouth_status == "OPEN"):
                    # Overlay the smoke animation on the frame at the appropriate location.
                    frame = overlay(frame, smoke_frame, face_landmarks, 
                                'MOUTH', mp_face_mesh.FACEMESH_LIPS)
                    
                if(left_eye_status == "OPEN"):
                    # Overlay the left eye image on the frame at the appropriate location.
                    frame = overlay(frame, left_eye, face_landmarks,
                                    'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE)
                   
                if(right_eye_status == "OPEN"):
                     # Overlay the right eye image on the frame at the appropriate location.
                    frame = overlay(frame, right_eye, face_landmarks,
                                'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE)
                    
        
        else:
            print("No face landmarks detected.")
            
        # SHOW THE IMAGE
        cv.imshow("Filtered Image", cv.flip(frame, 1))
        
        # BREAK THE LOOP
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    # RELEASE THE CAMERA AND DESTROY ALL WINDOWS
    cap.release()
    cv.destroyAllWindows()
