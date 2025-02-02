import requests
import csv
import time
import json

# The IP address of your ESP32 web server (Replace with your actual IP)
ESP32_IP = "http://192.168.10.161"  # Ensure the IP is correct for your network

# CSV file name to store the data
csv_filename = "sensor_data.csv"

# Function to get the sensor data from ESP32
def get_sensor_data():
    try:
        # Send an HTTP GET request to the ESP32 server with a timeout to avoid hanging indefinitely
        response = requests.get(ESP32_IP, timeout=10)

        # Log the raw response text to inspect
        print("Raw response:", response.text)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Clean the response text to avoid any extra characters or newlines
            raw_response = response.text.strip()  # Remove leading/trailing whitespaces/newlines
            data = json.loads(raw_response)  # Parse the cleaned-up JSON response

            # Extract the sensor values from the JSON response
            accel_x = data.get('accel_x')
            accel_y = data.get('accel_y')
            accel_z = data.get('accel_z')
            gyro_x = data.get('gyro_x')
            gyro_y = data.get('gyro_y')
            gyro_z = data.get('gyro_z')
            temperature = data.get('temp')

            # Return the data values
            return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature
        else:
            print(f"Failed to retrieve data from ESP32. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch any HTTP-related exceptions (timeouts, connection errors)
        print(f"Error fetching data: {e}")
        return None

# Function to write data to CSV
def write_to_csv(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature, timestamp):
    try:
        # Open the CSV file in append mode
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the sensor data along with the timestamp
            writer.writerow([timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature])
        print(f"Data saved to CSV at {timestamp}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Function to initialize the CSV file with a header if it doesn't exist
def initialize_csv():
    try:
        # Check if the CSV file exists and add the header if it's a new file
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Temperature"])
    except FileExistsError:
        # If the file exists, we don't need to write the header again
        pass

# Main function to fetch data and write it to CSV continuously
def main():
    # Initialize the CSV file with a header if it doesn't already exist
    initialize_csv()

    while True:
        # Get sensor data from ESP32
        sensor_data = get_sensor_data()
        if sensor_data:
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature = sensor_data

            # Get the current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

            # Print the fetched data (optional)
            print(f"Timestamp: {timestamp}, Accel_X: {accel_x}, Accel_Y: {accel_y}, Accel_Z: {accel_z}, "
                  f"Gyro_X: {gyro_x}, Gyro_Y: {gyro_y}, Gyro_Z: {gyro_z}, Temp: {temperature}")

            # Write the data to CSV
            write_to_csv(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature, timestamp)

        # Wait for a short period before fetching new data
        time.sleep(1)

if __name__ == "__main__":
    main()

