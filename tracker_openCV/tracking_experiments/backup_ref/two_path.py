import cv2
import numpy as np
import csv
import time

def initialize_kalman_filter():
    """
    Initializes and returns a Kalman filter for tracking an object.
    The filter tracks the position (x, y) and velocity (dx, dy).
    """
    kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy) and 2 measurements (x, y)

    # State transition matrix (defines how the state evolves)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

    # Measurement matrix (we measure x and y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

    # Process noise covariance (defines how much we expect the state to change)
    kalman.processNoiseCov = np.array([[1e-4, 0, 0, 0],
                                       [0, 1e-4, 0, 0],
                                       [0, 0, 1e-4, 0],
                                       [0, 0, 0, 1e-4]], np.float32)

    # Measurement noise covariance (defines expected measurement error)
    kalman.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], np.float32)

    # A priori error covariance (initial uncertainty in the state estimate)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman


def process_frame(frame, kalman_yellow, kalman_red):
    """
    Processes a single frame for detecting and tracking yellow and red objects.
    Applies Kalman filtering and returns the processed frame with object positions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color ranges for yellow and red objects
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
    red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))

    # Clean the masks using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours for yellow and red objects
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize object centers
    center_yellow = center_red = None
    predicted_yellow = predicted_red = None  # Initialize prediction variables

    # Process yellow object
    if contours_yellow:
        largest_yellow = max(contours_yellow, key=cv2.contourArea)
        x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(largest_yellow)
        center_yellow = (x_yellow + w_yellow // 2, y_yellow + h_yellow // 2)
        cv2.rectangle(frame, (x_yellow, y_yellow), (x_yellow + w_yellow, y_yellow + h_yellow), (0, 255, 0), 2)
        cv2.circle(frame, center_yellow, 5, (0, 0, 255), -1)

    # Process red object
    if contours_red:
        largest_red = max(contours_red, key=cv2.contourArea)
        x_red, y_red, w_red, h_red = cv2.boundingRect(largest_red)
        center_red = (x_red + w_red // 2, y_red + h_red // 2)
        cv2.rectangle(frame, (x_red, y_red), (x_red + w_red, y_red + h_red), (0, 0, 255), 2)
        cv2.circle(frame, center_red, 5, (255, 0, 0), -1)

    # Update Kalman filter and predict new positions
    if center_yellow:
        kalman_yellow.correct(np.array([[np.float32(center_yellow[0])], [np.float32(center_yellow[1])]]))
        predicted_yellow = kalman_yellow.predict()
        predicted_yellow = (int(predicted_yellow[0]), int(predicted_yellow[1]))
        cv2.circle(frame, predicted_yellow, 5, (0, 255, 255), -1)

    if center_red:
        kalman_red.correct(np.array([[np.float32(center_red[0])], [np.float32(center_red[1])]]))
        predicted_red = kalman_red.predict()
        predicted_red = (int(predicted_red[0]), int(predicted_red[1]))
        cv2.circle(frame, predicted_red, 5, (0, 0, 255), -1)  # Red predicted position

    return frame, center_yellow, center_red, predicted_yellow, predicted_red


def update_path(path, center, predicted_point):
    """
    Updates the actual and predicted paths with the current and predicted positions.
    """
    if center:
        path['actual'].append(center)
    if predicted_point:
        path['predicted'].append(predicted_point)


def save_to_csv(path_points, filename='objects_path_with_time.csv'):
    """
    Saves path points with timestamps to a CSV file.
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  # Write header if file is empty
            writer.writerow(['Time (s)', 'X (Yellow Actual)', 'Y (Yellow Actual)', 'X (Yellow Predicted)', 'Y (Yellow Predicted)',
                             'X (Red Actual)', 'Y (Red Actual)', 'X (Red Predicted)', 'Y (Red Predicted)'])
        for point in path_points:
            writer.writerow(point)

    print(f"Path points saved to '{filename}'.")


def track_objects(camera_index=0):
    """
    Main function to track yellow and red objects using Kalman filters.
    It updates paths, displays results, and saves the data to a CSV file.
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not access the camera at index {camera_index}.")
            return

        # Initialize Kalman filters for yellow and red objects
        kalman_yellow = initialize_kalman_filter()
        kalman_red = initialize_kalman_filter()

        # Store paths for both objects
        path_yellow = {'actual': [], 'predicted': []}
        path_red = {'actual': [], 'predicted': []}

        path_points_for_csv = []
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Process the frame and update paths
            frame, center_yellow, center_red, predicted_yellow, predicted_red = process_frame(
                frame, kalman_yellow, kalman_red)
            
            update_path(path_yellow, center_yellow, predicted_yellow)
            update_path(path_red, center_red, predicted_red)

            # Log data for CSV with timestamps
            elapsed_time = time.time() - start_time
            path_points_for_csv.append(
                (elapsed_time,
                 center_yellow[0] if center_yellow else None, center_yellow[1] if center_yellow else None,
                 predicted_yellow[0] if predicted_yellow else None, predicted_yellow[1] if predicted_yellow else None,
                 center_red[0] if center_red else None, center_red[1] if center_red else None,
                 predicted_red[0] if predicted_red else None, predicted_red[1] if predicted_red else None)
            )

            # Draw actual paths for yellow and red objects
            for i in range(1, len(path_yellow['actual'])):
                cv2.line(frame, path_yellow['actual'][i - 1], path_yellow['actual'][i], (255, 255, 204), 2)  # Light yellow for actual
            for i in range(1, len(path_red['actual'])):
                cv2.line(frame, path_red['actual'][i - 1], path_red['actual'][i], (0, 0, 255), 2)  # Red for actual

            # Draw predicted paths for yellow and red objects (Limit to last 5 points)
            if len(path_yellow['predicted']) > 0:
                # Limit the predicted path to the last 5 points
                for i in range(max(1, len(path_yellow['predicted']) - 5), len(path_yellow['predicted'])):
                    cv2.line(frame, path_yellow['predicted'][i - 1], path_yellow['predicted'][i], (0, 255, 255), 2)  # Yellow for predicted
            if len(path_red['predicted']) > 0:
                # Limit the predicted path to the last 5 points
                for i in range(max(1, len(path_red['predicted']) - 5), len(path_red['predicted'])):
                    cv2.line(frame, path_red['predicted'][i - 1], path_red['predicted'][i], (0, 0, 255), 2)  # Red for predicted

            # Display the processed frame
            cv2.imshow("Object Tracking", frame)

            # Exit condition on 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break

            # Save data to CSV when 's' key is pressed
            if key == ord('s'):
                save_to_csv(path_points_for_csv)

    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Start tracking objects
track_objects()

