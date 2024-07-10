import cv2
import numpy as np

# Define the range of the target color in HSV
lower_color1 = np.array([100, 150, 50])  # Lower bound of the color
upper_color1 = np.array([170, 250, 255])  # Upper bound of the color
# lower_color2 = np.array([95, 120, 0])
# upper_color2 = np.array([140, 255, 255])

background_image = cv2.imread('replacement_image_final.jpg')

if background_image is None:
    raise ValueError("Could not open or find the replacement image!")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the target color
    mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
    # mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
    mask = mask1 

    # Find contours of the masked color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(frame)
    
    for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                 
    # Convert the contour mask to grayscale
    contour_mask_gray = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)

    inverse_mask = cv2.bitwise_not(contour_mask_gray)

    background_region = cv2.bitwise_and(background_resized, background_resized, mask=contour_mask_gray)
    frame_region = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    combined = cv2.add(background_region, frame_region)

    # cv2.imshow('maskedImage', contour_mask)
    cv2.imshow('Frame', combined)
               
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()