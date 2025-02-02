import serial
import time

# Set up the serial port
SERIAL_PORT = '/dev/ttyUSB0'  # Change to your correct COM port, e.g., 'COM3' for Windows or '/dev/ttyUSB0' for Linux/Mac
BAUD_RATE = 115200  # Default baud rate for ESP8266
TIMEOUT = 1  # Timeout for serial communication

# Open the serial port
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
time.sleep(2)  # Wait for the ESP8266 to initialize

def send_command(command, delay=1):
    """Send AT command to ESP8266 and return the response"""
    print(f"Sending command: {command}")
    ser.write((command + "\r\n").encode())  # Send command with a newline character
    time.sleep(delay)  # Wait for response
    response = ser.read_all().decode('utf-8')
    print("Response:")
    print(response)
    return response

# Test the connection with AT commands
try:
    # Send AT command to check if the ESP8266 responds
    response = send_command('AT')
    if 'OK' in response:
        print("ESP8266 is responding to AT commands.")
    else:
        print("No response from ESP8266.")

    # Send AT+GMR to get the ESP8266 firmware version
    response = send_command('AT+GMR')
    print(f"ESP8266 Firmware Version:\n{response}")

    # Send AT+RST to reset the ESP8266
    response = send_command('AT+RST')
    print("ESP8266 reset command sent.")

    # Send AT+CWMODE=1 to set Wi-Fi mode to station (client) mode
    response = send_command('AT+CWMODE=1')
    print("Wi-Fi mode set to station mode.")

    # Send AT+CWJAP to connect to a Wi-Fi network
    wifi_ssid = 'IAAC-WIFI'
    wifi_password = 'EnterIaac22@'
    response = send_command(f'AT+CWJAP="{wifi_ssid}","{wifi_password}"')
    print("Attempting to connect to Wi-Fi.")

    # You can send more commands as needed:
    # response = send_command('AT+CIFSR')  # Get IP address

finally:
    # Close the serial connection
    ser.close()
    print("Serial connection closed.")

