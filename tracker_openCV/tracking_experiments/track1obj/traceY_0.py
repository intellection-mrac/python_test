import cv2
import numpy as np
import csv
import time

def initialize_kalman_filter():
    """
    Initializes the Kalman filter for object tracking.
    Returns the initialized Kalman filter object.
    """
    kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy) and 2 measurements (x, y)

    # State transition matrix
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

    # Measurement matrix (we only measure x and y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

    # Process noise covariance (system noise)
    kalman.processNoiseCov = np.array([[1e-4, 0, 0, 0],
                                       [0, 1e-4, 0, 0],
                                       [0, 0, 1e-4, 0],
                                       [0, 0, 0, 1e-4]], np.float32)

    # Measurement noise covariance (measurement error)
    kalman.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], np.float32)

    # A priori error covariance
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman

def track_yellow_object(camera_index=0):
    """
    Tracks a yellow object using a Kalman filter, and saves the current and predicted path to a CSV file.
    """
    try:
        # Open the USB camera (use the correct index)
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Error: Could not access the camera at index {camera_index}.")
            return  # Exit if the camera can't be accessed

        print(f"Camera initialized successfully at index {camera_index}. Press 'q' to quit and 's' to save the path points to CSV.")
        
        # Initialize Kalman Filter
        kalman = initialize_kalman_filter()

        # Lists to store coordinates for path drawing
        actual_path_points = []  # Actual path (measured positions)
        predicted_path_points = []  # Predicted path (Kalman filter predictions)

        # List to store timestamp, actual position, and predicted position for saving to CSV
        path_points_for_csv = []

        # Start time for timestamping
        start_time = time.time()

        while True:
            # Capture each frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break  # Exit the loop if a frame couldn't be captured

            # Convert the captured frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the range of yellow color in HSV space
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])

            # Create a mask for the yellow color
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Perform morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process the contours if found
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Draw a green bounding box around the yellow object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate the center of the object
                center = (x + w // 2, y + h // 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot at the center

                # Add the current position to the actual path
                actual_path_points.append(center)

                # Calculate time elapsed since the start
                elapsed_time = time.time() - start_time

                # Add timestamp and actual position to the list for CSV
                path_points_for_csv.append((elapsed_time, center[0], center[1], None, None))  # Actual position in CSV (no predicted values yet)

                # Update the Kalman filter with the current position (measurement)
                kalman.correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))

                # Predict the next position using the Kalman filter
                predicted = kalman.predict()

                # Extract scalar values and create the predicted point
                predicted_point = (int(predicted[0][0]), int(predicted[1][0]))  # Corrected line

                # Add the predicted point to the predicted path
                predicted_path_points.append(predicted_point)

                # Add timestamp and predicted position to the list for CSV
                path_points_for_csv[-1] = (elapsed_time, center[0], center[1], predicted_point[0], predicted_point[1])  # Update the CSV with predicted position

                # Draw the predicted path in yellow
                if len(predicted_path_points) > 1:
                    cv2.line(frame, predicted_path_points[-2], predicted_path_points[-1], (0, 255, 255), 2)  # Yellow path for predicted points

                # Draw the actual path in blue
                if len(actual_path_points) > 1:
                    cv2.line(frame, actual_path_points[-2], actual_path_points[-1], (255, 0, 0), 2)  # Blue path for actual points

                # Display the predicted position as a yellow circle
                cv2.circle(frame, predicted_point, 5, (0, 255, 255), -1)  # Yellow circle for predicted position

            # Display the original frame with tracking and prediction overlays
            cv2.imshow("Yellow Object Tracking and Prediction", frame)

            # Exit if the user presses 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break

            # Save path points to CSV if the user presses 's'
            if key == ord('s'):
                with open('yellow_object_path_with_time.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if csvfile.tell() == 0:  # Write header if the file is empty
                        writer.writerow(['Time (s)', 'X (Actual)', 'Y (Actual)', 'X (Predicted)', 'Y (Predicted)'])
                    for point in path_points_for_csv:
                        writer.writerow(point)
                print("Path points with time and predictions saved to 'yellow_object_path_with_time.csv'.")

    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Release the capture and close any open windows
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released and windows closed.")

if __name__ == "__main__":
    # Test USB camera with the correct index
    track_yellow_object(camera_index=0)  # Replace with your correct camera index

