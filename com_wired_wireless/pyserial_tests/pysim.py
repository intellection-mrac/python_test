import matplotlib.pyplot as plt
import csv
import time
from collections import deque
import threading
import random  # To simulate sensor data
import queue  # For thread-safe communication

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
    """Write collected data to CSV file."""
    try:
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
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Function to update the plot in real-time
def update_plot():
    """Update the plot with new data."""
    try:
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
    except Exception as e:
        print(f"Error updating plot: {e}")

# Function to simulate sensor data (accelerometer and gyroscope)
def simulate_sensor_data(q):
    """Simulate accelerometer and gyroscope data."""
    try:
        while True:
            # Simulate accelerometer data (3 values: X, Y, Z)
            acc = [random.uniform(-10, 10) for _ in range(3)]  # Random data between -10 and 10 m/s^2
            # Simulate gyroscope data (3 values: X, Y, Z)
            gyro = [random.uniform(-500, 500) for _ in range(3)]  # Random data between -500 and 500 degrees/s
            timestamp = time.time()

            # Store data in the queue for the main thread to process
            q.put((timestamp, acc, gyro))

            # Simulate a delay to mimic real-time data (e.g., every 0.1 seconds)
            time.sleep(0.1)

            # Simulate saving the data every 10 seconds
            if int(timestamp) % 10 == 0:  # Save data every 10 seconds
                save_to_csv()
                print(f"Data saved to CSV at timestamp {timestamp}")
    except Exception as e:
        print(f"Error in data simulation: {e}")

# Function to process the data from the queue and update the plot
def process_data(q):
    """Process data from the queue and update the plot."""
    global acc_data, gyro_data, timestamps
    try:
        while True:
            timestamp, acc, gyro = q.get()  # Wait for new data from the worker thread

            # Store data
            acc_data.append(acc)
            gyro_data.append(gyro)
            timestamps.append(timestamp)

            # Update the plot
            update_plot()
    except Exception as e:
        print(f"Error processing data: {e}")

# Main entry point for the program
def main():
    """Main function to initialize threads and start the simulation."""
    try:
        # Start the queue to communicate between threads
        q = queue.Queue()

        # Start data simulation and processing in separate threads
        simulation_thread = threading.Thread(target=simulate_sensor_data, args=(q,))
        simulation_thread.daemon = True  # Daemon thread will exit when the program ends
        simulation_thread.start()

        process_thread = threading.Thread(target=process_data, args=(q,))
        process_thread.daemon = True  # Daemon thread will exit when the program ends
        process_thread.start()

        # Show the plot (only in the main thread)
        plt.show()

        # Keep the program running to allow real-time updates
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Program terminated by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Run the main function
if __name__ == '__main__':
    main()

