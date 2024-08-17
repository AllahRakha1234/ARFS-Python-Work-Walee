import cv2
import dlib
import os

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
detector_path = os.path.join(os.getcwd(), 'models/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(detector_path)

# Load filter image (e.g., glasses) with transparency
filter_img = cv2.imread(os.path.join(os.getcwd(), r'data/testImages/heartGlasses.jpeg'), -1)

# Read image
img = cv2.imread(os.path.join(os.getcwd(), r'data/testImages/img2.jpeg'))

# Detect face and landmarks
faces = detector(img)
for face in faces:
    landmarks = predictor(img, face)
    
    # Get the coordinates of the eyes
    left_eye = landmarks.part(36)
    right_eye = landmarks.part(45)
    
    # Calculate the size and position of the filter
    filter_width = int((right_eye.x - left_eye.x) * 1.5)
    filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
    filter_x = int(left_eye.x - filter_width / 4)
    filter_y = int(left_eye.y - filter_height / 2)

    # Resize the filter to match the calculated dimensions
    resized_filter = cv2.resize(filter_img, (filter_width, filter_height), interpolation=cv2.INTER_AREA)

    # Extract the regions where the filter will be placed
    roi = img[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width]

    # Check if the resized filter has an alpha channel
    if resized_filter.shape[2] == 4:
        # Split the filter into its RGB and Alpha channels
        filter_rgb = resized_filter[:, :, :3]
        filter_alpha = resized_filter[:, :, 3]

        # Create a mask and its inverse from the alpha channel
        _, mask = cv2.threshold(filter_alpha, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
    else:
        # If no alpha channel, use the filter as is and create a mask from it
        filter_rgb = resized_filter
        filter_gray = cv2.cvtColor(resized_filter, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(filter_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

    # Ensure the mask and ROI sizes are the same
    if roi.shape[:2] != mask.shape[:2]:
        print("Size mismatch between ROI and filter mask")
        continue

    # Black-out the area of the filter in the ROI
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only the region of the filter from the filter image
    filter_fg = cv2.bitwise_and(filter_rgb, filter_rgb, mask=mask)

    # Add the filter to the ROI and modify the main image
    dst = cv2.add(img_bg, filter_fg)
    img[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width] = dst

# Display the output
cv2.imshow('Filtered Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
