import cv2
import numpy as np
import csv
import time

def track_yellow_object(camera_index=4):
    try:
        # Open the USB camera (use the correct index)
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Error: Could not access the camera at index {camera_index}.")
            return  # Exit if the camera can't be accessed

        print(f"Camera initialized successfully at index {camera_index}. Press 'q' to quit and 's' to save the path points to CSV.")
        
        # List to store coordinates for path drawing (just (x, y))
        path_points_for_drawing = []
        
        # List to store timestamp and coordinates for saving to CSV (time, x, y)
        path_points_for_csv = []

        # Start time (for time reference)
        start_time = time.time()

        while True:
            # Capture each frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break  # Exit the loop if a frame couldn't be captured

            # Convert the captured frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the range of the yellow color in HSV space
            lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
            upper_yellow = np.array([40, 255, 255])  # Upper bound for yellow

            # Create a mask for the yellow color
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Perform some morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If contours are found, process them
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
                center = (x + w // 2, y + h // 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot at the center

                # Add the coordinates to the list for drawing the path
                path_points_for_drawing.append(center)

                # Calculate time elapsed since the start
                elapsed_time = time.time() - start_time

                # Add timestamp and coordinates to the list for CSV
                path_points_for_csv.append((elapsed_time, center[0], center[1]))

                # Draw the path by connecting consecutive points
                for i in range(1, len(path_points_for_drawing)):
                    cv2.line(frame, path_points_for_drawing[i - 1], path_points_for_drawing[i], (255, 0, 0), 2)  # Blue path

            # Display the original frame with the tracking overlay
            cv2.imshow("Yellow Object Tracking", frame)

            # Break the loop if the user presses 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break  # Exit the loop if 'q' is pressed

            # Save the path points to CSV if the user presses 's'
            if key == ord('s'):
                with open('yellow_object_path_with_time.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Time (s)', 'X', 'Y'])  # Write header
                    for point in path_points_for_csv:
                        writer.writerow(point)
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

if __name__ == "__main__":
    # Test USB camera with the correct index
    track_yellow_object(camera_index=4)  # Replace with your correct camera index

