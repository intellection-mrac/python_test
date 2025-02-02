import serial
import matplotlib.pyplot as plt
import csv
import time
from collections import deque
import threading
import keyboard  # For capturing 's' key press

# Set up serial communication with the ESP module
esp_port = '/dev/ttyUSB0'  # Adjust as per your system
baud_rate = 115200
ser = serial.Serial(esp_port, baud_rate)

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

# Function to read data from the ESP module
def read_data():
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    # Assuming ESP sends data in the format:
                    # Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z
                    acc_data_values = list(map(float, line.split(',')))
                    acc = acc_data_values[:3]
                    gyro = acc_data_values[3:]
                    timestamp = time.time()

                    # Store data
                    acc_data.append(acc)
                    gyro_data.append(gyro)
                    timestamps.append(timestamp)

                    update_plot()
                except ValueError:
                    pass  # Ignore invalid data

# Function to listen for the 's' key to save the data
def listen_for_save():
    while True:
        if keyboard.is_pressed('s'):
            save_to_csv()
            print("Data saved to CSV.")
            time.sleep(1)  # To prevent multiple saves in a short time

# Start data reading and key listening in separate threads
thread1 = threading.Thread(target=read_data)
thread2 = threading.Thread(target=listen_for_save)
thread1.daemon = True
thread2.daemon = True
thread1.start()
thread2.start()

# Show the plot
plt.show()
