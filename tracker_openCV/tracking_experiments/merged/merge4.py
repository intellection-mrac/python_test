import cv2
import numpy as np
import time
import csv
import socket

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
csv_header = ["timestamp", "center_1_x", "center_1_y", "center_2_x", "center_2_y", 
              "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temperature",
              "accel_x_g", "accel_y_g", "accel_z_g", "velocity_x", "velocity_y", "velocity_z",
              "displacement_x", "displacement_y", "displacement_z", "filtered_accel_x", "filtered_accel_y", "filtered_accel_z"]

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

    if frame is None:
        print("Error: Frame is None.")
        return  # Prevent further processing if frame is None

    if event == cv2.EVENT_LBUTTONDOWN:
        if color_picker_state == 0:
            picked_color_1 = frame[y, x]
            print(f"Picked Color 1: {picked_color_1}")
            color_1_hsv_min, color_1_hsv_max = get_hsv_range(frame, x, y)
            color_picker_state = 1
            print("Color 1 selected, now pick Color 2.")
        elif color_picker_state == 1:
            picked_color_2 = frame[y, x]
            print(f"Picked Color 2: {picked_color_2}")
            color_2_hsv_min, color_2_hsv_max = get_hsv_range(frame, x, y)
            color_picker_state = 2
            print("Color 2 selected, ready to start tracking!")

# Function to calculate HSV range based on selected point
def get_hsv_range(frame, x, y, range_offset=10):
    """Calculates the HSV range around the picked color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_hsv = hsv[y, x]
    
    # Ensuring that hue is within the valid range [0, 179]
    lower_hue = (color_hsv[0] - range_offset) % 180  # Modulo ensures it wraps around the hue range
    upper_hue = (color_hsv[0] + range_offset) % 180  # Modulo ensures it wraps around the hue range

    # If the computed hue is out of bounds, apply modulus to keep it within [0, 179]
    lower_bound = np.array([lower_hue, 100, 100])
    upper_bound = np.array([upper_hue, 255, 255])
    
    return lower_bound, upper_bound

# Function to create mask for color tracking
def create_color_mask(hsv, color_min, color_max):
    """Create a binary mask for the object based on HSV color range."""
    mask = cv2.inRange(hsv, color_min, color_max)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Function to find the largest N contours in a mask
def find_largest_contours(mask, N=2):
    """Finds the largest N contours in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area
        return sorted_contours[:N]  # Return the largest N contours
    return []

# Function to receive MPU6050 data from ESP32 via socket
def receive_mpu6050_data():
    try:
        # Create a socket and connect to the ESP32
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('192.168.10.161', 8080))  # ESP32 IP and port
            data = s.recv(1024)  # Receive data (MPU6050 data)
            # Assuming the data is sent as a comma-separated string like:
            # "accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature"
            data = data.decode('utf-8')
            data = data.split(',')
            accel_x, accel_y, accel_z = map(float, data[:3])
            gyro_x, gyro_y, gyro_z = map(float, data[3:6])
            temperature = float(data[6])

            # Return sensor data
            return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature
    except Exception as e:
        print(f"Error receiving sensor data: {e}")
        return None, None, None, None, None, None, None

# Function to save tracking data to CSV
def save_tracking_data(centers_1, centers_2, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature):
    """Saves the tracking data to a CSV file."""
    timestamp = time.time()
    row = [timestamp]  # Start with the timestamp
    
    # Add center points for color 1
    if centers_1:
        row.extend([centers_1[0][0], centers_1[0][1]])  # Add center_1_x, center_1_y
    
    # Add center points for color 2
    if centers_2:
        row.extend([centers_2[0][0], centers_2[0][1]])  # Add center_2_x, center_2_y
    
    # Add sensor data
    row.extend([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature])

    # Save the row to CSV
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
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create masks for both selected colors
            mask_1 = create_color_mask(hsv, color_1_hsv_min, color_1_hsv_max)
            mask_2 = create_color_mask(hsv, color_2_hsv_min, color_2_hsv_max)

            # Find the largest N contours for both colors
            contours_1 = find_largest_contours(mask_1, N=2)
            contours_2 = find_largest_contours(mask_2, N=2)

            centers_1 = []
            centers_2 = []

            # Process first color (color 1)
            for contour_1 in contours_1:
                x_1, y_1, w_1, h_1 = cv2.boundingRect(contour_1)
                center_1 = (x_1 + w_1 // 2, y_1 + h_1 // 2)
                centers_1.append(center_1)
                cv2.rectangle(frame, (x_1, y_1), (x_1 + w_1, y_1 + h_1), (0, 0, 255), 2)
                cv2.circle(frame, center_1, 5, (255, 0, 0), -1)

            # Process second color (color 2)
            for contour_2 in contours_2:
                x_2, y_2, w_2, h_2 = cv2.boundingRect(contour_2)
                center_2 = (x_2 + w_2 // 2, y_2 + h_2 // 2)
                centers_2.append(center_2)
                cv2.rectangle(frame, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (255, 0, 0), 2)
                cv2.circle(frame, center_2, 5, (0, 0, 255), -1)

            # Receive MPU6050 data
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature = receive_mpu6050_data()

            # Save data to CSV
            save_tracking_data(centers_1, centers_2, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature)

        # Display the current frame
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start object tracking
track_objects()

