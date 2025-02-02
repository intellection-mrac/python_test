import requests
import time
import csv
import math
import sys

# The IP address of your ESP32 on the network
ESP32_IP = "http://192.168.215.244"  # Replace with your ESP32's actual IP address

# CSV file name
csv_filename = "sensor_data.csv"

# Function to get the sensor data from ESP32
def get_sensor_data():
    try:
        # Send an HTTP GET request to the ESP32 server
        response = requests.get(ESP32_IP)

        # Check if the request was successful (HTTP status 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Get the accelerometer, gyro, and temperature data
            accel_x = data['accel_x']
            accel_y = data['accel_y']
            accel_z = data['accel_z']
            gyro_x = data['gyro_x']
            gyro_y = data['gyro_y']
            gyro_z = data['gyro_z']
            temperature = data['temp']

            # Additional Calculations and Derived Data
            magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)  # Magnitude of acceleration
            g_force = magnitude / 9.81  # Normalize to g-force
            linear_acceleration_magnitude = magnitude - 9.81  # Subtract gravity (assuming stationary)
            movement_status = 'Moving' if magnitude > 0.1 else 'Stationary'  # Detect movement based on magnitude
            angular_velocity = math.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)  # Total angular velocity
            total_rotation = gyro_x + gyro_y + gyro_z  # Total rotation estimate

            # Get the current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

            return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature, magnitude, g_force, linear_acceleration_magnitude, movement_status, angular_velocity, total_rotation, timestamp
        else:
            print("Failed to retrieve data. Status code:", response.status_code)
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to export data to CSV
def export_to_csv(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature, magnitude, g_force, linear_acceleration_magnitude, movement_status, angular_velocity, total_rotation, timestamp):
    # Open the CSV file in append mode
    try:
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the sensor data along with the timestamp
            writer.writerow([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature, magnitude, g_force, linear_acceleration_magnitude, movement_status, angular_velocity, total_rotation])
        print(f"Data saved to {csv_filename}: {timestamp}, {accel_x}, {accel_y}, {accel_z}, {gyro_x}, {gyro_y}, {gyro_z}, {temperature}, {magnitude}, {g_force}, {linear_acceleration_magnitude}, {movement_status}, {angular_velocity}, {total_rotation}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Function to create a CSV file and write the header if it doesn't exist
def initialize_csv():
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Temperature", "Magnitude", "G_Force", "Linear_Acceleration_Magnitude", "Movement_Status", "Angular_Velocity", "Total_Rotation"])
    except FileExistsError:
        pass  # If the file exists, we don't need to write the header again

# Main function to continuously fetch data and export when 's' is pressed
def main():
    # Initialize CSV file and write header if needed
    initialize_csv()

    while True:
        # Get the sensor data
        sensor_data = get_sensor_data()
        if sensor_data:
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature, magnitude, g_force, linear_acceleration_magnitude, movement_status, angular_velocity, total_rotation, timestamp = sensor_data

            # Print the sensor data (optional)
            print(f"Timestamp: {timestamp}, Accel X: {accel_x}, Accel Y: {accel_y}, Accel Z: {accel_z}, Gyro X: {gyro_x}, Gyro Y: {gyro_y}, Gyro Z: {gyro_z}, Temp: {temperature}, Magnitude: {magnitude}, G-Force: {g_force}, Linear Acceleration Magnitude: {linear_acceleration_magnitude}, Movement Status: {movement_status}, Angular Velocity: {angular_velocity}, Total Rotation: {total_rotation}")

        # Check for user input ('s' to save data to CSV, 'q' to quit)
        user_input = input("Press 's' to save data, 'q' to quit: ").strip().lower()
        if user_input == 's':
            export_to_csv(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature, magnitude, g_force, linear_acceleration_magnitude, movement_status, angular_velocity, total_rotation, timestamp)
        elif user_input == 'q':
            print("Exiting program...")
            break  # Exit the loop and terminate the program
        else:
            print("Invalid input. Please press 's' to save data or 'q' to quit.")

        # Wait before getting new data
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

