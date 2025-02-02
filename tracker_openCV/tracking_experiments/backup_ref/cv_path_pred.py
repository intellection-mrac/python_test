import cv2
import numpy as np
import csv
import time
from collections import deque

# Function to track the yellow object and predict its path using simple prediction
# The function uses a USB camera, detects yellow objects in the video stream, 
# tracks their position, and predicts the future position. It displays both actual 
# and predicted paths as trails on the screen. The positions and times are saved 
# into a CSV file for analysis.
def track_yellow_object(camera_index=0):
    try:
        # Open the USB camera (use the correct index)
        cap = cv2.VideoCapture(camera_index)

        # Check if the camera was opened successfully
        if not cap.isOpened():
            print(f"Error: Could not access the camera at index {camera_index}.")
            return  # Exit if the camera can't be accessed

        print(f"Camera initialized successfully at index {camera_index}. Press 'q' to quit and 's' to save the path points to CSV.")
        
        # List to store coordinates for path drawing (just (x, y)) without maxlen to store trails
        actual_path_points = deque()  
        predicted_path_points = deque()

        # Start time (for time reference)
        start_time = time.time()

        while True:
            # Capture each frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break  # Exit the loop if a frame couldn't be captured

            # Convert the captured frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the range of the yellow color in HSV space
            lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
            upper_yellow = np.array([40, 255, 255])  # Upper bound for yellow

            # Create a mask to isolate the yellow color in the frame
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Perform some morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours of the yellow object in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If contours are found, process them to track the yellow object
            if contours:
                # Get the largest contour based on area
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Draw a bounding box around the detected yellow object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
                center = (x + w // 2, y + h // 2)  # Calculate the center of the object
                cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot at the center

                # Add the actual coordinates to the list for drawing the actual path
                actual_path_points.append(center)

                # Predict the next position (here we are using a simple prediction method)
                predicted_point = (center[0] + 5, center[1] + 5)  # Simple prediction (just adding 5 pixels)
                predicted_path_points.append(predicted_point)

                # Draw the trails for both the actual and predicted paths
                if len(actual_path_points) > 1:
                    for i in range(1, len(actual_path_points)):
                        cv2.line(frame, actual_path_points[i - 1], actual_path_points[i], (255, 0, 0), 2)  # Blue path for actual points

                if len(predicted_path_points) > 1:
                    for i in range(1, len(predicted_path_points)):
                        cv2.line(frame, predicted_path_points[i - 1], predicted_path_points[i], (0, 0, 255), 2)  # Red path for predicted points

                # Calculate time elapsed since the start
                elapsed_time = time.time() - start_time

                # Save the timestamp (in seconds) and position (X, Y) for actual and predicted positions
                with open('yellow_object_path_with_time.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([elapsed_time, center[0], center[1], elapsed_time, predicted_point[0], predicted_point[1]])

            # Display the original frame with the tracking overlay
            cv2.imshow("Yellow Object Tracking", frame)

            # Break the loop if the user presses 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break  # Exit the loop if 'q' is pressed

            # Save the path points to CSV if the user presses 's'
            if key == ord('s'):
                print("Saving path points...")
                with open('yellow_object_path_with_time.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Time (s)', 'Actual X', 'Actual Y', 'Predicted Time (s)', 'Predicted X', 'Predicted Y'])  # Write header
                    # Write both actual and predicted points
                    for i in range(len(actual_path_points)):
                        writer.writerow([elapsed_time, actual_path_points[i][0], actual_path_points[i][1], elapsed_time, predicted_path_points[i][0], predicted_path_points[i][1]])
                print("Path points with time saved to 'yellow_object_path_with_time.csv'.")

    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Release the capture and close any open windows
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released and windows closed.")

# Main entry point to test the camera and track the yellow object
if __name__ == "__main__":
    track_yellow_object(camera_index=0)  # Use camera index 0 (default camera)

