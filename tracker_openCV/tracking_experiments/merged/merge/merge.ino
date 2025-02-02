#include <Wire.h>
#include <WiFi.h>
#include <MPU6050.h>

// Define RGB LED pins
#define RED_PIN 19
#define GREEN_PIN 18
#define BLUE_PIN 17

// Create an MPU6050 object
MPU6050 mpu;

// Wi-Fi credentials
const char* ssid = "IAAC-WIFI";
const char* password = "EnterIaac22@";

// Timing for blinking the red LED when Wi-Fi is not connected
unsigned long previousMillis = 0;
const long blinkInterval = 1000; // Blink every second

void setup() {
  // Initialize Serial Monitor
  Serial.begin(115200);
  delay(1000);

  // Initialize LED pins as output
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);

  // Connect to Wi-Fi
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  // Initialize MPU6050
  Wire.begin();
  mpu.initialize();

  // Check for MPU6050 sensor connection
  if (mpu.testConnection()) {
    Serial.println("MPU6050 connection successful!");
  } else {
    Serial.println("MPU6050 connection failed. Red LED ON.");
    // Turn on Red LED to indicate sensor connection error
    digitalWrite(RED_PIN, HIGH);
  }
}

void loop() {
  // Handle Wi-Fi connection
  if (WiFi.status() != WL_CONNECTED) {
    unsigned long currentMillis = millis();

    // Blink the red LED when Wi-Fi is not connected
    if (currentMillis - previousMillis >= blinkInterval) {
      previousMillis = currentMillis;
      // Toggle the red LED state
      digitalWrite(RED_PIN, !digitalRead(RED_PIN));
    }

    Serial.println("Attempting to reconnect WiFi...");
    delay(1000);
  } else {
    // Wi-Fi connected, turn on the green LED
    digitalWrite(GREEN_PIN, HIGH);
    digitalWrite(RED_PIN, LOW);  // Turn off red LED when connected

    // Display IP address of the ESP32 in Serial Monitor
    Serial.print("WiFi Connected. IP Address: ");
    Serial.println(WiFi.localIP());
  }

  // Check if the sensor is still sending data
  if (mpu.testConnection()) {
    // Read data from the MPU6050 sensor
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    // Print the sensor data to the serial monitor
    Serial.print("Accelerometer X: "); Serial.print(ax);
    Serial.print(", Y: "); Serial.print(ay);
    Serial.print(", Z: "); Serial.println(az);
    Serial.print("Gyroscope X: "); Serial.print(gx);
    Serial.print(", Y: "); Serial.print(gy);
    Serial.print(", Z: "); Serial.println(gz);

    // Turn on blue LED when data is being sent
    digitalWrite(BLUE_PIN, HIGH);

    // Simulate sending data to Python (Add your communication code here)

    // After sending data, turn off blue LED
    digitalWrite(BLUE_PIN, LOW);
  } else {
    // Sensor is not responding, turn on red LED to indicate error
    digitalWrite(RED_PIN, HIGH);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(BLUE_PIN, LOW);
  }

  delay(100); // Small delay before next loop iteration
}

