import cv2
import dlib
import os

# LOAD FACE DETECTOR AND LANDMARK PREDICTOR
detector = dlib.get_frontal_face_detector()
detector_path = os.path.join(os.getcwd(), 'models/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(detector_path)

# LOAD FILTER IMAGE (E.G., GLASSES) WITH TRANSPARENCY
filter_img = cv2.imread(os.path.join(os.getcwd(), r'data/testImages/roundGlassesTr.png'), -1)

# READ IMAGE
img = cv2.imread(os.path.join(os.getcwd(), r'data/testImages/img2.jpeg'))

# DETECT FACE AND LANDMARKS
faces = detector(img)
for face in faces:
    landmarks = predictor(img, face)
    
    # GET THE COORDINATES OF THE EYES
    left_eye = landmarks.part(36)
    right_eye = landmarks.part(45)
    
    # CALCULATE THE SIZE AND POSITION OF THE FILTER
    filter_width = int((right_eye.x - left_eye.x) * 1.5)
    filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
    filter_x = int(left_eye.x - filter_width / 4)
    filter_y = int(left_eye.y - filter_height / 2)

    # RESIZE THE FILTER TO MATCH THE CALCULATED DIMENSIONS
    resized_filter = cv2.resize(filter_img, (filter_width, filter_height), interpolation=cv2.INTER_AREA)

    # EXTRACT THE REGIONS WHERE THE FILTER WILL BE PLACED
    roi = img[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width]

    # CHECK IF THE RESIZED FILTER HAS AN ALPHA CHANNEL
    if resized_filter.shape[2] == 4:
        # SPLIT THE FILTER INTO ITS RGB AND ALPHA CHANNELS
        filter_rgb = resized_filter[:, :, :3]
        filter_alpha = resized_filter[:, :, 3]

        # CREATE A MASK AND ITS INVERSE FROM THE ALPHA CHANNEL
        _, mask = cv2.threshold(filter_alpha, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
    else:
        # IF NO ALPHA CHANNEL, USE THE FILTER AS IS AND CREATE A MASK FROM IT
        filter_rgb = resized_filter
        filter_gray = cv2.cvtColor(resized_filter, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(filter_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

    # ENSURE THE MASK AND ROI SIZES ARE THE SAME
    if roi.shape[:2] != mask.shape[:2]:
        print("SIZE MISMATCH BETWEEN ROI AND FILTER MASK")
        continue

    # BLACK-OUT THE AREA OF THE FILTER IN THE ROI
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # TAKE ONLY THE REGION OF THE FILTER FROM THE FILTER IMAGE
    filter_fg = cv2.bitwise_and(filter_rgb, filter_rgb, mask=mask)

    # ADD THE FILTER TO THE ROI AND MODIFY THE MAIN IMAGE
    dst = cv2.add(img_bg, filter_fg)
    img[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width] = dst

# DISPLAY THE OUTPUT
cv2.imshow('Filtered Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
