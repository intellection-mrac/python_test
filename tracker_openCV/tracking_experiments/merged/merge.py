import cv2
import numpy as np
import time
import csv
import socket
from datetime import datetime
from smbus import SMBus

# Configuration Variables
ESP32_IP_ADDRESS = "192.168.10.161"  # Modify this with the ESP32 IP address

# MPU6050 I2C address
MPU6050_ADDR = 0x68

# Camera index and other initial settings
CAMERA_INDEX = 4  # Camera index for the connected camera

# Initial HSV range values
color_1_hsv_min = np.array([0, 0, 0])
color_1_hsv_max = np.array([179, 255, 255])

color_2_hsv_min = np.array([0, 0, 0])
color_2_hsv_max = np.array([179, 255, 255])

# CSV file for tracking
csv_file = "tracking_data.csv"
csv_header = ["timestamp", "center_1_x", "center_1_y", "center_2_x", "center_2_y",
              "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
              "temperature", "accel_magnitude", "gyro_magnitude", "pitch", "roll", "yaw",
              "accel_x_g", "accel_y_g", "accel_z_g", "velocity_x", "velocity_y", "velocity_z",
              "displacement_x", "displacement_y", "displacement_z",
              "accel_raw_x", "accel_raw_y", "accel_raw_z", "gyro_raw_x", "gyro_raw_y", "gyro_raw_z",
              "filtered_accel_x", "filtered_accel_y", "filtered_accel_z"]

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

# Low-pass filter for accelerometer data (to reduce noise)
def low_pass_filter(data, prev_data, alpha=0.8):
    """Applies a low-pass filter to smooth accelerometer data."""
    return alpha * data + (1 - alpha) * prev_data

# MPU6050 data acquisition (accelerometer and gyroscope)
def read_mpu6050_data(bus, address=MPU6050_ADDR):
    """Reads accelerometer and gyroscope data from the MPU6050."""
    # Read accelerometer values
    accel_data = bus.read_i2c_block_data(address, 0x3B, 6)
    accel_x = np.int16(accel_data[0] << 8 | accel_data[1])
    accel_y = np.int16(accel_data[2] << 8 | accel_data[3])
    accel_z = np.int16(accel_data[4] << 8 | accel_data[5])

    # Read gyroscope values
    gyro_data = bus.read_i2c_block_data(address, 0x43, 6)
    gyro_x = np.int16(gyro_data[0] << 8 | gyro_data[1])
    gyro_y = np.int16(gyro_data[2] << 8 | gyro_data[3])
    gyro_z = np.int16(gyro_data[4] << 8 | gyro_data[5])

    # Read temperature
    temp_data = bus.read_i2c_block_data(address, 0x41, 2)
    temp_raw = np.int16(temp_data[0] << 8 | temp_data[1])
    temperature = (temp_raw / 340.0) + 36.53  # Celsius

    return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature

# Function to compute pitch, roll, and yaw from accelerometer and gyroscope
def compute_orientation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt):
    """Computes pitch, roll, and yaw angles from accelerometer and gyroscope data."""
    # Pitch and roll from accelerometer data
    pitch = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2))
    roll = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2))

    # Simple gyroscope integration for yaw (assuming small angle approximation)
    yaw = gyro_z * dt  # Assuming we are integrating over time with dt as timestep

    return np.degrees(pitch), np.degrees(roll), np.degrees(yaw)

# Function to save tracking data to CSV
def save_tracking_data(centers_1, centers_2, accel_data, gyro_data, temperature, timestamp, prev_accel):
    """Saves the tracking data to a CSV file."""
    accel_x, accel_y, accel_z = accel_data
    gyro_x, gyro_y, gyro_z = gyro_data

    # Calculate accelerometer magnitudes
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

    # Convert to g (assuming full-scale range of ±2g)
    accel_x_g = accel_x / 16384.0
    accel_y_g = accel_y / 16384.0
    accel_z_g = accel_z / 16384.0

    # Compute velocity (placeholder, assuming constant velocity)
    velocity_x, velocity_y, velocity_z = 0, 0, 0  # For simplicity, update this with real computation

    # Compute displacement (placeholder, assuming no displacement)
    displacement_x, displacement_y, displacement_z = 0, 0, 0

    # Apply low-pass filter to smooth accelerometer data
    filtered_accel_x = low_pass_filter(accel_x, prev_accel[0])
    filtered_accel_y = low_pass_filter(accel_y, prev_accel[1])
    filtered_accel_z = low_pass_filter(accel_z, prev_accel[2])

    # Calculate orientation (pitch, roll, yaw)
    pitch, roll, yaw = compute_orientation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, 0.1)

    # Save data to CSV
    row = [timestamp, *centers_1, *centers_2, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
           temperature, accel_magnitude, gyro_magnitude, pitch, roll, yaw, accel_x_g, accel_y_g, accel_z_g,
           velocity_x, velocity_y, velocity_z, displacement_x, displacement_y, displacement_z,
           accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, filtered_accel_x, filtered_accel_y, filtered_accel_z]
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# Main function to capture frames, track objects, and allow color picking
def track_objects(camera_index=CAMERA_INDEX):
    """Main function to track two colors selected by the user."""
    global color_1_hsv_min, color_1_hsv_max, color_2_hsv_min, color_2_hsv_max
    bus = SMBus(1)  # Initialize I2C bus (usually 1 on Raspberry Pi)
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    prev_accel = [0, 0, 0]  # Placeholder for previous accelerometer data

    # Start capturing video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        timestamp = time.time()  # Timestamp for each frame

        # Get sensor data from MPU6050
        accel_data = read_mpu6050_data(bus)
        gyro_data = accel_data[3:6]
        temperature = accel_data[6]

        # Save the data to CSV
        save_tracking_data([0, 0], [0, 0], accel_data[:3], gyro_data, temperature, timestamp, prev_accel)

        prev_accel = accel_data[:3]  # Update previous accelerometer data for filtering

        # Show the frame with the tracking results
        cv2.imshow("Object Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

        # Example interaction with ESP32 using the IP Address
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((ESP32_IP_ADDRESS, 80))  # Connect to ESP32 IP and Port 80 (HTTP)
                s.sendall(b'GET /data HTTP/1.1\r\n')
                response = s.recv(1024)
                print("ESP32 Response:", response.decode('utf-8'))
        except Exception as e:
            print(f"Error connecting to ESP32 at {ESP32_IP_ADDRESS}: {e}")

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting object tracking and data logging...")
    track_objects(camera_index=CAMERA_INDEX)

