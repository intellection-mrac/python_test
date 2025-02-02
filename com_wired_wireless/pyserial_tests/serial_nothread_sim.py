import serial
import matplotlib.pyplot as plt
import csv
import time
from collections import deque
from pynput import keyboard  # Using pynput for key press detection
import random  # For generating simulated data

# Set up serial communication (this part is just for structure, not used in simulation)
esp_port = '/dev/ttyUSB0'  # Adjust as per your system
baud_rate = 115200
# ser = serial.Serial(esp_port, baud_rate)  # Not needed for simulation

# Set up the plot window
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Accelerometer Data')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (m/s^2)')
ax2.set_title('Gyroscope Data')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angular velocity (deg/s)')

# Data buffers to store incoming data
acc_data = deque(maxlen=100)
gyro_data = deque(maxlen=100)
timestamps = deque(maxlen=100)

# Data file for exporting
csv_filename = 'sensor_data.csv'
fieldnames = ['Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# Function to write data to CSV
def save_to_csv():
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:  # Write header if the file is empty
            writer.writeheader()
        for timestamp, acc, gyro in zip(timestamps, acc_data, gyro_data):
            writer.writerow({
                'Timestamp': timestamp,
                'Accel_X': acc[0],
                'Accel_Y': acc[1],
                'Accel_Z': acc[2],
                'Gyro_X': gyro[0],
                'Gyro_Y': gyro[1],
                'Gyro_Z': gyro[2],
            })

# Function to update the plot in real-time
def update_plot():
    ax1.clear()
    ax2.clear()
    ax1.plot(timestamps, [acc[0] for acc in acc_data], label='Accel_X')
    ax1.plot(timestamps, [acc[1] for acc in acc_data], label='Accel_Y')
    ax1.plot(timestamps, [acc[2] for acc in acc_data], label='Accel_Z')
    ax2.plot(timestamps, [gyro[0] for gyro in gyro_data], label='Gyro_X')
    ax2.plot(timestamps, [gyro[1] for gyro in gyro_data], label='Gyro_Y')
    ax2.plot(timestamps, [gyro[2] for gyro in gyro_data], label='Gyro_Z')
    ax1.legend()
    ax2.legend()
    plt.draw()
    plt.pause(0.1)

# Function to detect key presses
def on_press(key):
    try:
        if key.char == 's':
            save_to_csv()
            print("Data saved to CSV.")
    except AttributeError:
        pass  # Handle special keys like Shift, Ctrl, etc.

# Main loop for simulating data and saving to CSV
with keyboard.Listener(on_press=on_press) as listener:
    while True:
        # Simulate accelerometer and gyroscope data (random values)
        acc = [random.uniform(-10, 10) for _ in range(3)]  # Simulated accelerometer data
        gyro = [random.uniform(-500, 500) for _ in range(3)]  # Simulated gyroscope data
        timestamp = time.time()

        # Store data
        acc_data.append(acc)
        gyro_data.append(gyro)
        timestamps.append(timestamp)

        update_plot()

        # Keep listening for key events
        listener.join()  # This will block the loop and listen for key presses

        # Optionally, exit the loop after a condition (e.g., a key press or timeout)
        # if keyboard.is_pressed('q'):  # Uncomment this if you want to quit on 'q'
        #     break

