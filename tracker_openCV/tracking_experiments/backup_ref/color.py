import cv2
import numpy as np
import csv
import time


def initialize_kalman_filter():
    """
    Initializes and returns a Kalman filter for tracking an object.
    Tracks the position (x, y) and velocity (dx, dy).
    """
    kalman = cv2.KalmanFilter(4, 2)  # 4 states: x, y, dx, dy; 2 measurements: x, y

    # State transition matrix (defines how the state evolves)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

    # Measurement matrix (we measure x and y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

    # Process noise covariance (how much we expect the state to change)
    kalman.processNoiseCov = np.array([[1e-2, 0, 0, 0],
                                       [0, 1e-2, 0, 0],
                                       [0, 0, 1e-2, 0],
                                       [0, 0, 0, 1e-2]], np.float32)

    # Measurement noise covariance (measurement error)
    kalman.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], np.float32)

    # A priori error covariance (initial uncertainty in state estimate)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    return kalman


def smooth_detection(current, previous):
    """
    Smooths the detected center using a simple moving average.
    """
    if previous is None:
        return current
    return (int((current[0] + previous[0]) / 2), int((current[1] + previous[1]) / 2))


def process_frame(frame, kalman_color1, kalman_color2, prev_center_color1, prev_center_color2):
    """
    Processes a single frame, detects two target colors, applies Kalman filtering,
    and returns the processed frame with object positions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for the two new colors based on the updated HSV ranges (including darker and lighter ranges)
    color1_lower = np.array([39, 120, 180])  # Lower HSV for color near (59, 205, 251)
    color1_upper = np.array([79, 255, 255])  # Upper HSV for color near (59, 205, 251)
    
    # Expand the range further for color1 to account for darker/lighter shades
    color1_lower = np.array([max(0, color1_lower[0] - 10), max(0, color1_lower[1] - 50), max(0, color1_lower[2] - 50)])  # Darker shade
    color1_upper = np.array([min(179, color1_upper[0] + 10), min(255, color1_upper[1] + 50), min(255, color1_upper[2] + 50)])  # Lighter shade

    color2_lower = np.array([310, 180, 200])  # Lower HSV for color near (329, 244, 236)
    color2_upper = np.array([350, 255, 255])  # Upper HSV for color near (329, 244, 236)

    # Expand the range further for color2 to account for darker/lighter shades
    color2_lower = np.array([max(0, color2_lower[0] - 10), max(0, color2_lower[1] - 50), max(0, color2_lower[2] - 50)])  # Darker shade
    color2_upper = np.array([min(179, color2_upper[0] + 10), min(255, color2_upper[1] + 50), min(255, color2_upper[2] + 50)])  # Lighter shade

    mask_color1 = cv2.inRange(hsv, color1_lower, color1_upper)
    mask_color2 = cv2.inRange(hsv, color2_lower, color2_upper)

    # Combine the two masks
    combined_mask = cv2.bitwise_or(mask_color1, mask_color2)

    # Perform morphological operations to clean the masks
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours for the two color objects
    contours_color1, _ = cv2.findContours(mask_color1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_color2, _ = cv2.findContours(mask_color2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center_color1 = center_color2 = None
    predicted_color1 = predicted_color2 = None

    # Process color1 object
    if contours_color1:
        largest_color1 = max(contours_color1, key=cv2.contourArea)
        x_color1, y_color1, w_color1, h_color1 = cv2.boundingRect(largest_color1)
        center_color1 = (x_color1 + w_color1 // 2, y_color1 + h_color1 // 2)
        center_color1 = smooth_detection(center_color1, prev_center_color1)
        cv2.rectangle(frame, (x_color1, y_color1), (x_color1 + w_color1, y_color1 + h_color1), (0, 255, 0), 2)
        cv2.circle(frame, center_color1, 5, (0, 0, 255), -1)

    # Process color2 object
    if contours_color2:
        largest_color2 = max(contours_color2, key=cv2.contourArea)
        x_color2, y_color2, w_color2, h_color2 = cv2.boundingRect(largest_color2)
        center_color2 = (x_color2 + w_color2 // 2, y_color2 + h_color2 // 2)
        center_color2 = smooth_detection(center_color2, prev_center_color2)
        cv2.rectangle(frame, (x_color2, y_color2), (x_color2 + w_color2, y_color2 + h_color2), (0, 0, 255), 2)
        cv2.circle(frame, center_color2, 5, (255, 0, 0), -1)

    # Update Kalman filters and predict new positions
    if center_color1:
        kalman_color1.correct(np.array([[np.float32(center_color1[0])], [np.float32(center_color1[1])]]))
        predicted_color1 = kalman_color1.predict()
        predicted_color1 = (int(predicted_color1[0][0]), int(predicted_color1[1][0]))
        cv2.circle(frame, predicted_color1, 5, (0, 255, 255), -1)

    if center_color2:
        kalman_color2.correct(np.array([[np.float32(center_color2[0])], [np.float32(center_color2[1])]]))
        predicted_color2 = kalman_color2.predict()
        predicted_color2 = (int(predicted_color2[0][0]), int(predicted_color2[1][0]))
        cv2.circle(frame, predicted_color2, 5, (0, 0, 255), -1)

    return frame, center_color1, center_color2, predicted_color1, predicted_color2


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
    Saves the path points with timestamps to a CSV file.
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  # Write header if the file is empty
            writer.writerow(['Time (s)', 'X (Color1 Actual)', 'Y (Color1 Actual)', 'X (Color1 Predicted)', 'Y (Color1 Predicted)',
                             'X (Color2 Actual)', 'Y (Color2 Actual)', 'X (Color2 Predicted)', 'Y (Color2 Predicted)'])
        for point in path_points:
            writer.writerow(point)

    print(f"Path points saved to '{filename}'.")


def track_objects(camera_index=4):  # Change the camera index to 4
    """
    Main function to track the two new colors using Kalman filters.
    It updates paths, displays results, and saves the data to a CSV file.
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not access the camera at index {camera_index}.")
            return

        # Initialize Kalman filters for both colors
        kalman_color1 = initialize_kalman_filter()
        kalman_color2 = initialize_kalman_filter()

        # Store paths for both color objects
        path_color1 = {'actual': [], 'predicted': []}
        path_color2 = {'actual': [], 'predicted': []}

        path_points_for_csv = []
        start_time = time.time()
        prev_center_color1 = None
        prev_center_color2 = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Process the frame and update paths
            frame, center_color1, center_color2, predicted_color1, predicted_color2 = process_frame(
                frame, kalman_color1, kalman_color2, prev_center_color1, prev_center_color2)

            update_path(path_color1, center_color1, predicted_color1)
            update_path(path_color2, center_color2, predicted_color2)

            # Log data for CSV with timestamps
            elapsed_time = time.time() - start_time
            path_points_for_csv.append(
                (elapsed_time,
                 center_color1[0] if center_color1 else None, center_color1[1] if center_color1 else None,
                 predicted_color1[0] if predicted_color1 else None, predicted_color1[1] if predicted_color1 else None,
                 center_color2[0] if center_color2 else None, center_color2[1] if center_color2 else None,
                 predicted_color2[0] if predicted_color2 else None, predicted_color2[1] if predicted_color2 else None)
            )

            # Update previous centers for smoothing
            prev_center_color1 = center_color1
            prev_center_color2 = center_color2

            # Draw actual paths for both color objects
            for i in range(1, len(path_color1['actual'])):
                cv2.line(frame, path_color1['actual'][i - 1], path_color1['actual'][i], (255, 255, 204), 2)
            for i in range(1, len(path_color2['actual'])):
                cv2.line(frame, path_color2['actual'][i - 1], path_color2['actual'][i], (0, 0, 255), 2)

            # Draw predicted paths for both color objects (Limit to last 5 points)
            if len(path_color1['predicted']) > 0:
                for i in range(max(1, len(path_color1['predicted']) - 5), len(path_color1['predicted'])):
                    cv2.line(frame, path_color1['predicted'][i - 1], path_color1['predicted'][i], (0, 255, 255), 2)
            if len(path_color2['predicted']) > 0:
                for i in range(max(1, len(path_color2['predicted']) - 5), len(path_color2['predicted'])):
                    cv2.line(frame, path_color2['predicted'][i - 1], path_color2['predicted'][i], (0, 0, 255), 2)

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
        cap.release()  # Release the camera resource
        cv2.destroyAllWindows()  # Close any OpenCV windows


# Start object tracking
track_objects(camera_index=4)  # Camera index is set to 4

