import cv2
import numpy as np

# Load the image
image = cv2.imread('1.jpg')

# Convert the image from BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the red color in HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# Create a mask that identifies the red color
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# You can also use another range for red color if it's in the higher hue range
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# Combine the two masks to cover all ranges of red
full_mask = cv2.bitwise_or(mask, mask2)

# Use the mask to extract the red areas from the image
red_areas = cv2.bitwise_and(image, image, mask=full_mask)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Red Areas', red_areas)
cv2.imshow('Mask', full_mask)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

