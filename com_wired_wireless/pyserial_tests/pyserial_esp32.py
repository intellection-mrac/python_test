import serial
import time

# Set the correct port for your system
# On Windows, you might use something like 'COM3'
# On Linux/Mac, you might use '/dev/ttyUSB0' or '/dev/ttyACM0'
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Adjust the COM port if needed

def read_data():
    while True:
        if ser.in_waiting > 0:
            # Read a line of data from the serial port
            data = ser.readline().decode('utf-8').strip()  # Decode from bytes to string
            if data:  # Only print if data is not empty
                print(data)

if __name__ == '__main__':
    try:
        print("Reading accelerometer and gyroscope data from ESP32...")
        time.sleep(2)  # Wait for the ESP32 to initialize
        
        # Read and print accelerometer and gyroscope data continuously
        read_data()

    except KeyboardInterrupt:
        print("Exiting...")
        ser.close()  # Close the serial connection when done

