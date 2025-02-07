import cv2
import numpy as np

# Global variables to store the selected colors
selected_color_1 = None
selected_color_2 = None
color_count = 0  # Variable to keep track of color selection

# Mouse callback function to select colors
def select_color(event, x, y, flags, param):
    global selected_color_1, selected_color_2, color_count, image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the color of the clicked pixel (in BGR format)
        selected_color = image[y, x]
        print(f"Selected color: {selected_color}")

        # If the first color hasn't been selected, save it
        if color_count == 0:
            selected_color_1 = selected_color
            color_count = 1
            print("First color selected. Click to select the second color.")
        
        # If the first color has been selected, save the second color
        elif color_count == 1:
            selected_color_2 = selected_color
            color_count = 2
            print("Second color selected.")
        
        # If both colors are selected, process the image
        if color_count == 2:
            process_image()

# Process the image to display selected colors on a white background
def process_image():
    global selected_color_1, selected_color_2, image

    # Create a white background (same size as the original image)
    white_background = np.ones_like(image) * 255

    # Create masks for both selected colors (with a tolerance of 20)
    lower_bound_1 = np.array([max(c - 20, 0) for c in selected_color_1])  # Tolerance for first color
    upper_bound_1 = np.array([min(c + 20, 255) for c in selected_color_1])

    lower_bound_2 = np.array([max(c - 20, 0) for c in selected_color_2])  # Tolerance for second color
    upper_bound_2 = np.array([min(c + 20, 255) for c in selected_color_2])

    # Apply the masks to extract the colors
    mask_1 = cv2.inRange(image, lower_bound_1, upper_bound_1)
    mask_2 = cv2.inRange(image, lower_bound_2, upper_bound_2)

    # Create the final output image
    result_image = white_background.copy()

    # Set first color (black) in the image where the first mask is active
    result_image[mask_1 > 0] = [0, 0, 0]

    # Set second color (gray) where the second mask is active
    result_image[mask_2 > 0] = [169, 169, 169]  # Gray color

    # Convert the image to grayscale for the output
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Display the processed image
    cv2.imshow('Processed Image', gray_image)

    # Save the processed image
    cv2.imwrite('processed_image.jpg', gray_image)
    print("Processed image saved as 'processed_image.jpg'.")

# Load the image
image_path = '2.jpg'  # Update with your image path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load the image. Please check the file path.")
else:
    # Display the original image
    cv2.imshow('Original Image', image)

    # Set the mouse callback function for selecting colors
    cv2.setMouseCallback('Original Image', select_color)

    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

