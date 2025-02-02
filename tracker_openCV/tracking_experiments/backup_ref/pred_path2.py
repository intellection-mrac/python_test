import cv2
import numpy as np
import csv
import time
from collections import deque

# Function to track yellow object and predict its path
def track_yellow_object(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not access camera at index {camera_index}.")
        return

    print("Camera initialized. Press 'q' to quit and 's' to save the path points.")

    actual_path = deque(maxlen=500)  # Keep a long trail for actual path
    predicted_path = deque(maxlen=5)  # Keep a smaller trail for predicted path

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the yellow color range
        mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))

        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w // 2, y + h // 2)

            # Update paths
            actual_path.append(center)
            predicted_point = (center[0] + 5, center[1] + 5)
            predicted_path.append(predicted_point)

            # Draw bounding box and points
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)  # Actual position in yellow
            cv2.circle(frame, predicted_point, 5, (0, 128, 128), -1)  # Predicted position in teal

            # Draw paths
            for i in range(1, len(actual_path)):
                cv2.line(frame, actual_path[i - 1], actual_path[i], (255, 255, 180), 2)  # Light yellow path

            for i in range(1, len(predicted_path)):
                cv2.line(frame, predicted_path[i - 1], predicted_path[i], (0, 128, 128), 2)  # Teal path

            # Log time and positions
            elapsed_time = time.time() - start_time
            with open('yellow_object_path_with_time.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([elapsed_time, center[0], center[1], predicted_point[0], predicted_point[1]])

        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        if key == ord('s'):  # Save path to CSV
            with open('yellow_object_path_with_time.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Time (s)', 'Actual X', 'Actual Y', 'Predicted X', 'Predicted Y'])
                for actual, predicted in zip(actual_path, predicted_path):
                    writer.writerow([elapsed_time, actual[0], actual[1], predicted[0], predicted[1]])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_yellow_object(camera_index=0)

