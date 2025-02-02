import cv2
import numpy as np
import time
import csv

# Camera index and other initial settings
CAMERA_INDEX = 4  # Camera index for the connected camera

# Variables for color picking and the initial HSV range values
color_picker_state = 0  # 0 - No color picked, 1 - Color 1 picked, 2 - Color 2 picked
picked_color_1 = None
picked_color_2 = None
color_1_hsv_min = np.array([0, 0, 0])
color_1_hsv_max = np.array([179, 255, 255])
color_2_hsv_min = np.array([0, 0, 0])
color_2_hsv_max = np.array([179, 255, 255])

# Instructions for the color picker
COLOR_PICKER_INSTRUCTION = "Click to select Color 1. Once done, click for Color 2."

# CSV file for tracking with descriptive headers
CSV_FILE = "tracking_data.csv"
CSV_HEADERS = [
    "Timestamp", 
    "Color_1_Center_1_X", "Color_1_Center_1_Y", "Color_1_Center_2_X", "Color_1_Center_2_Y",
    "Color_2_Center_1_X", "Color_2_Center_1_Y", "Color_2_Center_2_X", "Color_2_Center_2_Y"
]

# Function to initialize Kalman filter
def initialize_kalman_filter():
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_hsv = hsv[y, x]
    
    lower_hue = (color_hsv[0] - range_offset) % 180  # Modulo ensures it wraps around the hue range
    upper_hue = (color_hsv[0] + range_offset) % 180  # Modulo ensures it wraps around the hue range

    lower_bound = np.array([lower_hue, 100, 100])
    upper_bound = np.array([upper_hue, 255, 255])
    
    return lower_bound, upper_bound

# Function to create mask for color tracking
def create_color_mask(hsv, color_min, color_max):
    mask = cv2.inRange(hsv, color_min, color_max)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Function to find the largest N contours in a mask
def find_largest_contours(mask, N=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area
        return sorted_contours[:N]  # Return the largest N contours
    return []

# Function to apply Kalman filtering for smoother tracking
def smooth_detection(current, previous, alpha=0.7):
    if previous is None:
        return current
    return (int(alpha * current[0] + (1 - alpha) * previous[0]),
            int(alpha * current[1] + (1 - alpha) * previous[1]))

# Function to process frames, track objects, and update Kalman filters
def process_frame(frame, kalman_1, kalman_2, prev_center_1, prev_center_2):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for both selected colors
    mask_1 = create_color_mask(hsv, color_1_hsv_min, color_1_hsv_max)
    mask_2 = create_color_mask(hsv, color_2_hsv_min, color_2_hsv_max)

    # Find the largest N contours for both colors
    contours_1 = find_largest_contours(mask_1, N=2)
    contours_2 = find_largest_contours(mask_2, N=2)

    centers_1 = []
    centers_2 = []
    predicted_1 = predicted_2 = None

    # Process first color (color 1)
    for contour_1 in contours_1:
        x_1, y_1, w_1, h_1 = cv2.boundingRect(contour_1)
        center_1 = (x_1 + w_1 // 2, y_1 + h_1 // 2)
        centers_1.append(center_1)
        center_1 = smooth_detection(center_1, prev_center_1)
        cv2.rectangle(frame, (x_1, y_1), (x_1 + w_1, y_1 + h_1), (0, 0, 255), 2)
        cv2.circle(frame, center_1, 5, (255, 0, 0), -1)

    # Process second color (color 2)
    for contour_2 in contours_2:
        x_2, y_2, w_2, h_2 = cv2.boundingRect(contour_2)
        center_2 = (x_2 + w_2 // 2, y_2 + h_2 // 2)
        centers_2.append(center_2)
        center_2 = smooth_detection(center_2, prev_center_2)
        cv2.rectangle(frame, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (255, 0, 0), 2)
        cv2.circle(frame, center_2, 5, (0, 0, 255), -1)

    # Update Kalman filters and predict new positions
    if len(centers_1) > 0:
        kalman_1.correct(np.array([[np.float32(centers_1[0][0])], [np.float32(centers_1[0][1])]]))
        predicted_1 = kalman_1.predict()
        predicted_1 = (int(predicted_1[0][0]), int(predicted_1[1][0]))
        cv2.circle(frame, predicted_1, 5, (0, 255, 255), -1)

    if len(centers_2) > 0:
        kalman_2.correct(np.array([[np.float32(centers_2[0][0])], [np.float32(centers_2[0][1])]]))
        predicted_2 = kalman_2.predict()
        predicted_2 = (int(predicted_2[0][0]), int(predicted_2[1][0]))
        cv2.circle(frame, predicted_2, 5, (0, 255, 255), -1)

    return frame, centers_1, centers_2, predicted_1, predicted_2

# Function to save tracking data to CSV
def save_tracking_data(centers_1, centers_2):
    timestamp = time.time()
    row = [timestamp]
    
    # Fill in center coordinates for the two colors (ensure we track up to 2 objects for each color)
    for i in range(2):  # For Color 1 (up to 2 objects)
        if i < len(centers_1):
            row.extend([centers_1[i][0], centers_1[i][1]])
        else:
            row.extend([None, None])  # If there's no second object, fill with None
    
    for i in range(2):  # For Color 2 (up to 2 objects)
        if i < len(centers_2):
            row.extend([centers_2[i][0], centers_2[i][1]])
        else:
            row.extend([None, None])  # If there's no second object, fill with None
    
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# Main function to capture frames, track objects, and allow color picking
def track_objects(camera_index=CAMERA_INDEX):
    global color_picker_state, color_1_hsv_min, color_1_hsv_max, color_2_hsv_min, color_2_hsv_max

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Initialize Kalman filters for both colors
    kalman_1 = initialize_kalman_filter()
    kalman_2 = initialize_kalman_filter()

    prev_center_1 = prev_center_2 = None

    # Write headers to CSV if the file is empty
    try:
        with open(CSV_FILE, mode='r', newline='') as file:
            file.seek(0, 2)  # Move to the end of the file
            if file.tell() == 0:  # If the file is empty
                with open(CSV_FILE, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(CSV_HEADERS)
    except FileNotFoundError:
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)

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
            frame, centers_1, centers_2, predicted_1, predicted_2 = process_frame(frame, kalman_1, kalman_2, prev_center_1, prev_center_2)
            prev_center_1 = centers_1[0] if centers_1 else None
            prev_center_2 = centers_2[0] if centers_2 else None

            # Save tracking data to CSV
            save_tracking_data(centers_1, centers_2)

            # Show the frame with the tracking results
            cv2.imshow("Object Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

        # Add functionality to reset tracking and save the data
        if key == ord('r'):  # Reset button (Press 'r' to reset and restart tracking)
            print("Resetting and saving data...")
            save_tracking_data(centers_1, centers_2)  # Export CSV before resetting
            color_picker_state = 0  # Reset color picker state
            print("Colors reset, please pick new colors.")

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_objects(camera_index=CAMERA_INDEX)

