import requests
import json

# The IP address of your ESP32 web server
ESP32_IP = "http://192.168.10.161"  # Replace with the actual IP address

# Function to get sensor data
def get_sensor_data():
    try:
        # Send a GET request to the ESP32 server
        response = requests.get(ESP32_IP)

        # Check if the response status is OK (200)
        if response.status_code == 200:
            # Clean up any unwanted characters (like \r\n)
            response_text = response.text.strip()  # Remove any unwanted newlines or spaces

            # Parse the JSON response
            data = json.loads(response_text)

            # Extract and return the data
            return data['accel_x'], data['accel_y'], data['accel_z'], data['gyro_x'], data['gyro_y'], data['gyro_z'], data['temp']
        else:
            print(f"Error: Unable to fetch data (Status Code: {response.status_code})")
            return None

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Main loop to fetch and display sensor data
while True:
    sensor_data = get_sensor_data()
    if sensor_data:
        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp = sensor_data
        print(f"Accel_X: {accel_x}, Accel_Y: {accel_y}, Accel_Z: {accel_z}, "
              f"Gyro_X: {gyro_x}, Gyro_Y: {gyro_y}, Gyro_Z: {gyro_z}, Temp: {temp}")
    else:
        print("No data retrieved.")

    # Wait before fetching new data
    time.sleep(1)

