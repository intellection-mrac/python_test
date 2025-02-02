import cv2
import numpy as np
import time
import csv

# Camera index and other initial settings
CAMERA_INDEX = 4  # Camera index for the connected camera

# Initial HSV range values
color_1_hsv_min = np.array([0, 0, 0])
color_1_hsv_max = np.array([179, 255, 255])

color_2_hsv_min = np.array([0, 0, 0])
color_2_hsv_max = np.array([179, 255, 255])

# Variables for color picking
color_picker_state = 0  # 0 - No color picked, 1 - Color 1 picked, 2 - Color 2 picked
picked_color_1 = None
picked_color_2 = None

# Instructions for the color picker
COLOR_PICKER_INSTRUCTION = "Click to select Color 1. Once done, click for Color 2."

# Global CSV file for tracking
csv_file = "tracking_data.csv"
csv_header = ["timestamp", "color_1_hsv_min", "color_1_hsv_max", "color_2_hsv_min", "color_2_hsv_max", 
              "center_1_x", "center_1_y", "center_2_x", "center_2_y"]

# Initialize Kalman filter
def initialize_kalman_filter():
    """Initializes a Kalman filter for object tracking."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.array([[1e-2, 0, 0, 0],
                                       [0, 1e-2, 0, 0],
                                       [0, 0, 1e-2, 0],
                                       [0, 0, 0, 1e-2]], np.float32)
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman

# Mouse callback function to handle color picking
def pick_color(event, x, y, flags, param):
    """Color picker callback for mouse click events."""
    global color_picker_state, picked_color_1, picked_color_2, color_1_hsv_min, color_1_hsv_max, color_2_hsv_min, color_2_hsv_max
    frame = param  # Access the frame from param argument

    if event == cv2.EVENT_LBUTTONDOWN:
        if color_picker_state == 0:
            picked_color_1 = frame[y, x]
            print(f"Picked Color 1: {picked_color_1}")
            color_1_hsv_min, color_1_hsv_max = get_hsv_range(frame, x, y)
            color_picker_state = 1
            print("Color 1 selected, now pick Color 2.")
            cv2.putText(frame, "Color 1 Selected. Now pick Color 2.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif color_picker_state == 1:
            picked_color_2 = frame[y, x]
            print(f"Picked Color 2: {picked_color_2}")
            color_2_hsv_min, color_2_hsv_max = get_hsv_range(frame, x, y)
            color_picker_state = 2
            print("Color 2 selected, ready to start tracking!")
            cv2.putText(frame, "Both Colors Selected! Starting Tracking...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Function to calculate HSV range based on selected point
def get_hsv_range(frame, x, y, range_offset=10):
    """Calculates the HSV range around the picked color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_hsv = hsv[y, x]
    lower_bound = np.array([max(0, color_hsv[0] - range_offset), 100, 100])
    upper_bound = np.array([min(179, color_hsv[0] + range_offset), 255, 255])
    return lower_bound, upper_bound

# Function to create mask for color tracking
def create_color_mask(hsv, color_min, color_max):
    """Create a binary mask for the object based on HSV color range."""
    mask = cv2.inRange(hsv, color_min, color_max)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Function to find the largest contour in a mask
def find_largest_contour(mask):
    """Finds the largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

# Function to apply Kalman filtering for smoother tracking
def smooth_detection(current, previous, alpha=0.7):
    """Applies exponential smoothing to the current detection."""
    if previous is None:
        return current
    return (int(alpha * current[0] + (1 - alpha) * previous[0]),
            int(alpha * current[1] + (1 - alpha) * previous[1]))

# Function to process frames, track objects, and update Kalman filters
def process_frame(frame, kalman_1, kalman_2, prev_center_1, prev_center_2):
    """Processes the frame and tracks the two colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for both selected colors
    mask_1 = create_color_mask(hsv, color_1_hsv_min, color_1_hsv_max)
    mask_2 = create_color_mask(hsv, color_2_hsv_min, color_2_hsv_max)

    # Find contours for both colors
    contour_1 = find_largest_contour(mask_1)
    contour_2 = find_largest_contour(mask_2)

    center_1 = center_2 = None
    predicted_1 = predicted_2 = None

    # Process first color (color 1)
    if contour_1 is not None:
        x_1, y_1, w_1, h_1 = cv2.boundingRect(contour_1)
        center_1 = (x_1 + w_1 // 2, y_1 + h_1 // 2)
        center_1 = smooth_detection(center_1, prev_center_1)
        cv2.rectangle(frame, (x_1, y_1), (x_1 + w_1, y_1 + h_1), (0, 0, 255), 2)
        cv2.circle(frame, center_1, 5, (255, 0, 0), -1)

    # Process second color (color 2)
    if contour_2 is not None:
        x_2, y_2, w_2, h_2 = cv2.boundingRect(contour_2)
        center_2 = (x_2 + w_2 // 2, y_2 + h_2 // 2)
        center_2 = smooth_detection(center_2, prev_center_2)
        cv2.rectangle(frame, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (255, 0, 0), 2)
        cv2.circle(frame, center_2, 5, (0, 0, 255), -1)

    # Update Kalman filters and predict new positions
    if center_1:
        kalman_1.correct(np.array([[np.float32(center_1[0])], [np.float32(center_1[1])]]))
        predicted_1 = kalman_1.predict()
        predicted_1 = (int(predicted_1[0][0]), int(predicted_1[1][0]))
        cv2.circle(frame, predicted_1, 5, (0, 255, 255), -1)

    if center_2:
        kalman_2.correct(np.array([[np.float32(center_2[0])], [np.float32(center_2[1])]]))
        predicted_2 = kalman_2.predict()
        predicted_2 = (int(predicted_2[0][0]), int(predicted_2[1][0]))
        cv2.circle(frame, predicted_2, 5, (0, 255, 255), -1)

    return frame, center_1, center_2, predicted_1, predicted_2

# Function to save tracking data to CSV
def save_tracking_data(center_1, center_2):
    """Saves the tracking data to a CSV file."""
    timestamp = time.time()
    row = [timestamp, str(color_1_hsv_min), str(color_1_hsv_max), str(color_2_hsv_min), str(color_2_hsv_max),
           center_1[0] if center_1 else None, center_1[1] if center_1 else None,
           center_2[0] if center_2 else None, center_2[1] if center_2 else None]
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# Main function to capture frames, track objects, and allow color picking
def track_objects(camera_index=CAMERA_INDEX):
    """Main function to track two colors selected by the user."""
    global color_picker_state, color_1_hsv_min, color_1_hsv_max, color_2_hsv_min, color_2_hsv_max

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Initialize Kalman filters for both colors
    kalman_1 = initialize_kalman_filter()
    kalman_2 = initialize_kalman_filter()

    prev_center_1 = prev_center_2 = None

    # Start capturing video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Set up the color picker
        if color_picker_state < 2:
            cv2.putText(frame, COLOR_PICKER_INSTRUCTION, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Color Picker", frame)
            cv2.setMouseCallback("Color Picker", pick_color, frame)

        # If both colors are picked, start tracking them
        if color_picker_state == 2:
            frame, center_1, center_2, predicted_1, predicted_2 = process_frame(frame, kalman_1, kalman_2, prev_center_1, prev_center_2)
            prev_center_1 = center_1
            prev_center_2 = center_2

            # Save tracking data to CSV
            save_tracking_data(center_1, center_2)

            # Show the frame with the tracking results
            cv2.imshow("Object Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

        # Add functionality to reset tracking and save the data
        if key == ord('r'):  # Reset button (Press 'r' to reset and restart tracking)
            print("Resetting and saving data...")
            save_tracking_data(center_1, center_2)  # Export CSV before resetting
            color_picker_state = 0  # Reset color picker state
            print("Colors reset, please pick new colors.")

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_objects(camera_index=CAMERA_INDEX)

