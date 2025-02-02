import requests
import time
import csv
import math
import sys
import platform

# For cross-platform key press detection
if platform.system() == "Windows":
    import msvcrt  # For Windows
else:
    import tty
    import termios

# ESP32 IP Address and Port
ESP32_IP = "http://192.168.215.244"  # Replace with your ESP32's actual IP address
PORT = 80  # Default HTTP port

# CSV file to store sensor data
csv_filename = "sensor_data.csv"

# Function to calculate vibration magnitude
def calculate_vibration_magnitude(accel_x, accel_y, accel_z):
    return math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

# Function to calculate jerk (change in acceleration)
def calculate_jerk(accel_x, accel_y, accel_z, prev_accel):
    jerk_x = accel_x - prev_accel[0]
    jerk_y = accel_y - prev_accel[1]
    jerk_z = accel_z - prev_accel[2]
    return jerk_x, jerk_y, jerk_z

# Function to apply temperature compensation to sensor data
def apply_temperature_compensation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp):
    temp_offset = temp - 25.0  # Offset from 25°C
    accel_x -= temp_offset * 0.01
    accel_y -= temp_offset * 0.01
    accel_z -= temp_offset * 0.01
    gyro_x -= temp_offset * 0.1
    gyro_y -= temp_offset * 0.1
    gyro_z -= temp_offset * 0.1
    return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

# Function to export data to CSV
def export_to_csv(data, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    print(f"Data saved to {filename}: {data}")

# Function to read data from ESP32 via HTTP request
def get_sensor_data():
    try:
        # Make an HTTP GET request to the ESP32
        response = requests.get(f"{ESP32_IP}:{PORT}/")

        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            return data
        else:
            print(f"Error: Unable to retrieve data. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to detect key press (cross-platform)
def key_pressed():
    if platform.system() == "Windows":
        return msvcrt.kbhit() and msvcrt.getch().decode('utf-8').lower() == 's'
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            if sys.stdin.read(1) == 's':
                return True
            return False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# Main function to handle data collection and CSV export
def main():
    # Create the CSV file and write the header if it doesn't exist
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Temperature", "Vibration_Magnitude", "Jerk_X", "Jerk_Y", "Jerk_Z"])
    except FileExistsError:
        pass  # If the file exists, we don't need to write the header again

    prev_accel = [0.0, 0.0, 0.0]  # Previous accelerometer values for jerk calculation

    while True:
        # Get sensor data from the ESP32
        data = get_sensor_data()

        if data:
            accel_x = data["accel_x"]
            accel_y = data["accel_y"]
            accel_z = data["accel_z"]
            gyro_x = data["gyro_x"]
            gyro_y = data["gyro_y"]
            gyro_z = data["gyro_z"]
            temp = data["temperature"]

            # Apply temperature compensation
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = apply_temperature_compensation(
                accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp)

            # Calculate vibration magnitude
            vibration_magnitude = calculate_vibration_magnitude(accel_x, accel_y, accel_z)

            # Calculate jerk (change in acceleration)
            jerk_x, jerk_y, jerk_z = calculate_jerk(accel_x, accel_y, accel_z, prev_accel)

            # Get current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

            # Prepare data to be written to CSV
            row = [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp, vibration_magnitude, jerk_x, jerk_y, jerk_z]

            # Print data (optional)
            print(f"Timestamp: {timestamp}, Accel X: {accel_x}, Accel Y: {accel_y}, Accel Z: {accel_z}, "
                  f"Gyro X: {gyro_x}, Gyro Y: {gyro_y}, Gyro Z: {gyro_z}, Temp: {temp}, "
                  f"Vibration Magnitude: {vibration_magnitude}, Jerk X: {jerk_x}, Jerk Y: {jerk_y}, Jerk Z: {jerk_z}")

            # Check if 's' key is pressed to save data to CSV
            if key_pressed():
                export_to_csv(row, csv_filename)

            # Update previous accelerometer values for jerk calculation
            prev_accel = [accel_x, accel_y, accel_z]

        # Wait before fetching data again
        time.sleep(1)

if __name__ == "__main__":
    main()

