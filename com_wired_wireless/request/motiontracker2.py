import cv2
import numpy as np
import time
import csv
import requests

# Camera index and HSV ranges for color detection
CAMERA_INDEX = 0
PINK_HSV_MIN = np.array([167, 149, 182])  # Adjusted for lumo pink
PINK_HSV_MAX = np.array([187, 249, 255])
YELLOW_HSV_MIN = np.array([25, 100, 100])  # Adjusted for lumo yellow
YELLOW_HSV_MAX = np.array([35, 255, 255])

# CSV file for data logging
csv_file = "tracking_data.csv"
csv_header = [
    "timestamp", 
    "chisel_pink_1_x", "chisel_pink_1_y", 
    "chisel_pink_2_x", "chisel_pink_2_y",
    "hammer_yellow_x", "hammer_yellow_y",
    "chisel_pink_1_confidence", "chisel_pink_2_confidence", 
    "hammer_yellow_confidence",
    "gyro_x", "gyro_y", "gyro_z", 
    "accel_x", "accel_y", "accel_z"
]

# ESP32 server settings
ESP32_IP = "192.168.10.161"
ESP32_PORT = 80
ESP32_URL = f"http://{ESP32_IP}:{ESP32_PORT}/"

# Function to initialize video capture
def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit(1)
    print("Camera initialized successfully.")
    return cap

# Function to create a binary mask for color detection
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
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return sorted_contours[:N]
    return []

# Function to fetch MPU6050 data from ESP32
def fetch_mpu6050_data():
    try:
        response = requests.get(ESP32_URL, timeout=1)
        if response.status_code == 200:
            data = response.json()
            print(f"ESP32 Data Fetched: {data}")
            return (
                data.get("gyro_x", 0.0), data.get("gyro_y", 0.0), data.get("gyro_z", 0.0),
                data.get("accel_x", 0.0), data.get("accel_y", 0.0), data.get("accel_z", 0.0)
            )
        else:
            print(f"Error: ESP32 returned status code {response.status_code}")
            return None, None, None, None, None, None
    except Exception as e:
        print(f"Error fetching ESP32 data: {e}")
        return None, None, None, None, None, None

# Function to log data to CSV
def log_to_csv(timestamp, chisel_centers, hammer_center, gyro_data, accel_data):
    row = [timestamp]
    # Chisel marker data
    if chisel_centers and len(chisel_centers) == 2:
        row.extend([
            chisel_centers[0][0] if chisel_centers[0] else None,
            chisel_centers[0][1] if chisel_centers[0] else None,
            chisel_centers[1][0] if chisel_centers[1] else None,
            chisel_centers[1][1] if chisel_centers[1] else None,
            1 if chisel_centers[0] else 0,
            1 if chisel_centers[1] else 0
        ])
    else:
        row.extend([None, None, None, None, 0, 0])
    # Hammer marker data
    if hammer_center:
        row.extend([hammer_center[0], hammer_center[1], 1])
    else:
        row.extend([None, None, 0])
    # MPU6050 data
    row.extend([
        gyro_data[0], gyro_data[1], gyro_data[2],
        accel_data[0], accel_data[1], accel_data[2]
    ])
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    print(f"Data logged: {row}")

# Main tracking function
def track_motion():
    cap = initialize_camera(CAMERA_INDEX)

    # Write CSV header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

    print("Press 's' to start tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        cv2.imshow("Preview (Press 's')", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Tracking started!")
            break

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect markers
        mask_pink = create_color_mask(hsv, PINK_HSV_MIN, PINK_HSV_MAX)
        mask_yellow = create_color_mask(hsv, YELLOW_HSV_MIN, YELLOW_HSV_MAX)

        pink_contours = find_largest_contours(mask_pink, N=2)
        yellow_contours = find_largest_contours(mask_yellow, N=1)

        pink_marker_1, pink_marker_2 = None, None
        hammer_marker = None

        if len(pink_contours) > 0:
            x, y, w, h = cv2.boundingRect(pink_contours[0])
            pink_marker_1 = (x + w // 2, y + h // 2)
        if len(pink_contours) > 1:
            x, y, w, h = cv2.boundingRect(pink_contours[1])
            pink_marker_2 = (x + w // 2, y + h // 2)

        if len(yellow_contours) > 0:
            x, y, w, h = cv2.boundingRect(yellow_contours[0])
            hammer_marker = (x + w // 2, y + h // 2)

        timestamp = time.time()
        gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z = fetch_mpu6050_data()

        # Log and visualize
        log_to_csv(
            timestamp, 
            [pink_marker_1, pink_marker_2], 
            hammer_marker, 
            [gyro_x, gyro_y, gyro_z], 
            [accel_x, accel_y, accel_z]
        )

        if pink_marker_1:
            cv2.circle(frame, pink_marker_1, 5, (255, 0, 255), -1)
        if pink_marker_2:
            cv2.circle(frame, pink_marker_2, 5, (255, 0, 255), -1)
        if hammer_marker:
            cv2.circle(frame, hammer_marker, 5, (0, 255, 255), -1)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tracking stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the program
track_motion()
