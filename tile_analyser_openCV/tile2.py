import cv2
import numpy as np

# Initialize global variables for selected pixel color
selected_hue = None
selected_color = None

# Define adjustable threshold values (Hue, Saturation, Value)
hue_threshold = 10  # This will adjust the range of hue (in degrees, 0-179)
saturation_threshold = 50  # Lower bound of saturation range
value_threshold = 50  # Lower bound of value range

# Mouse callback function to capture a color on click
def pick_color(event, x, y, flags, param):
    global selected_color, selected_hue
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR color of the pixel at the click position
        selected_color = image[y, x]
        
        # Convert the selected BGR color to HSV
        hsv_color = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)
        selected_hue = hsv_color[0][0][0]  # Get the Hue of the color

        print(f"Selected Color (BGR): {selected_color}")
        print(f"Selected Color (HSV): {hsv_color}")

# Load the image
image = cv2.imread('1.jpg')

# Display the image and set up the mouse callback function
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', pick_color)

while True:
    # Wait for a key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

    # If a color has been selected, process the image
    if selected_color is not None:
        # Convert the image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the tolerance for the selected color (in HSV)
        # Hue threshold defines how much variation in the hue is allowed
        lower_hue = np.clip(selected_hue - hue_threshold, 0, 179)
        upper_hue = np.clip(selected_hue + hue_threshold, 0, 179)

        # Saturation and Value thresholds will allow us to isolate specific brightness/strength
        lower_bound = np.array([lower_hue, saturation_threshold, value_threshold])
        upper_bound = np.array([upper_hue, 255, 255])

        # Create the mask for the selected color range
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Display the result with the mask applied
        cv2.imshow('Masked Image', result)

# Close all OpenCV windows
cv2.destroyAllWindows()

