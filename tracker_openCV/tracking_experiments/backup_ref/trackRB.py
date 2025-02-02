import cv2
import numpy as np
import time

# Define variables for camera settings
CAMERA_INDEX = 4  # Camera index
EXPOSURE_VALUE = -1  # Default auto exposure, or set a custom value (e.g., -4, -2, 0)

# Define initial HSV ranges for red and blue (to be calibrated dynamically)
RED_HSV_MIN_1 = np.array([0, 100, 100])
RED_HSV_MAX_1 = np.array([10, 255, 255])

RED_HSV_MIN_2 = np.array([170, 100, 100])
RED_HSV_MAX_2 = np.array([180, 255, 255])

BLUE_HSV_MIN = np.array([100, 150, 50])
BLUE_HSV_MAX = np.array([140, 255, 255])


def initialize_kalman_filter():
    """Initializes a Kalman filter for object tracking."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.array([[1e-2, 0, 0, 0],
                                       [0, 1e-2, 0, 0],
                                       [0, 0, 1e-2, 0],
                                       [0, 0, 0, 1e-2]], np.float32)
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


def create_color_mask(hsv, color_min, color_max):
    """Create a binary mask for the object based on HSV color range."""
    mask = cv2.inRange(hsv, color_min, color_max)
    kernel = np.ones((5, 5), np.uint8)  # Change kernel size for better performance if needed
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_largest_contour(mask):
    """Finds the largest contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None


def smooth_detection(current, previous, alpha=0.7):
    """Applies exponential smoothing to the current detection."""
    if previous is None:
        return current
    return (int(alpha * current[0] + (1 - alpha) * previous[0]),
            int(alpha * current[1] + (1 - alpha) * previous[1]))


def adjust_camera_exposure(cap, exposure_value=EXPOSURE_VALUE):
    """Adjust camera exposure (if supported)."""
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)  # -1 for auto, or set a value to adjust
        print(f"Camera exposure set to: {exposure_value}")


def process_frame(frame, kalman_red, kalman_blue, prev_center_red, prev_center_blue, red_hue_range, blue_hue_range):
    """Process each frame and track the red and blue objects using Kalman filters."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Dynamically calibrate red and blue color ranges based on the current frame
    red_min_1 = np.array([red_hue_range[0], 100, 100])
    red_max_1 = np.array([red_hue_range[1], 255, 255])
    
    blue_min = np.array([blue_hue_range[0], 150, 50])
    blue_max = np.array([blue_hue_range[1], 255, 255])

    # Create masks for red and blue objects
    red_mask_1 = create_color_mask(hsv, red_min_1, red_max_1)
    red_mask_2 = create_color_mask(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    blue_mask = create_color_mask(hsv, blue_min, blue_max)

    # Find contours
    red_contour = find_largest_contour(red_mask)
    blue_contour = find_largest_contour(blue_mask)

    center_red = center_blue = None
    predicted_red = predicted_blue = None

    # Process red object
    if red_contour is not None:
        x_red, y_red, w_red, h_red = cv2.boundingRect(red_contour)
        center_red = (x_red + w_red // 2, y_red + h_red // 2)
        center_red = smooth_detection(center_red, prev_center_red)
        cv2.rectangle(frame, (x_red, y_red), (x_red + w_red, y_red + h_red), (0, 0, 255), 2)
        cv2.circle(frame, center_red, 5, (255, 0, 0), -1)

    # Process blue object
    if blue_contour is not None:
        x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(blue_contour)
        center_blue = (x_blue + w_blue // 2, y_blue + h_blue // 2)
        center_blue = smooth_detection(center_blue, prev_center_blue)
        cv2.rectangle(frame, (x_blue, y_blue), (x_blue + w_blue, y_blue + h_blue), (255, 0, 0), 2)
        cv2.circle(frame, center_blue, 5, (0, 0, 255), -1)

    # Update Kalman filters and predict new positions
    if center_red:
        kalman_red.correct(np.array([[np.float32(center_red[0])], [np.float32(center_red[1])]]))
        predicted_red = kalman_red.predict()
        predicted_red = (int(predicted_red[0][0]), int(predicted_red[1][0]))
        cv2.circle(frame, predicted_red, 5, (0, 255, 255), -1)

    if center_blue:
        kalman_blue.correct(np.array([[np.float32(center_blue[0])], [np.float32(center_blue[1])]]))
        predicted_blue = kalman_blue.predict()
        predicted_blue = (int(predicted_blue[0][0]), int(predicted_blue[1][0]))
        cv2.circle(frame, predicted_blue, 5, (0, 255, 255), -1)

    return frame, center_red, center_blue, predicted_red, predicted_blue


def track_objects(camera_index=CAMERA_INDEX):
    """Main function to track red and blue objects, with dynamic color calibration."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Initialize Kalman filters for red and blue
    kalman_red = initialize_kalman_filter()
    kalman_blue = initialize_kalman_filter()

    prev_center_red = prev_center_blue = None
    start_time = time.time()

    # Default hue ranges (may be dynamically calibrated during the loop)
    red_hue_range = (0, 10)
    blue_hue_range = (100, 140)

    # Set up GUI window and trackbars
    cv2.namedWindow("HSV Calibration")
    
    # Red Hue Range Trackbars
    cv2.createTrackbar('Red Min Hue', 'HSV Calibration', red_hue_range[0], 179, lambda x: None)
    cv2.createTrackbar('Red Max Hue', 'HSV Calibration', red_hue_range[1], 179, lambda x: None)
    
    # Blue Hue Range Trackbars
    cv2.createTrackbar('Blue Min Hue', 'HSV Calibration', blue_hue_range[0], 179, lambda x: None)
    cv2.createTrackbar('Blue Max Hue', 'HSV Calibration', blue_hue_range[1], 179, lambda x: None)

    # Exposure Control Trackbar
    cv2.createTrackbar('Exposure', 'HSV Calibration', 0, 100, lambda x: None)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Get current trackbar positions for HSV ranges and exposure
            red_min_hue = cv2.getTrackbarPos('Red Min Hue', 'HSV Calibration')
            red_max_hue = cv2.getTrackbarPos('Red Max Hue', 'HSV Calibration')
            
            blue_min_hue = cv2.getTrackbarPos('Blue Min Hue', 'HSV Calibration')
            blue_max_hue = cv2.getTrackbarPos('Blue Max Hue', 'HSV Calibration')

            # Get current exposure setting
            exposure_value = cv2.getTrackbarPos('Exposure', 'HSV Calibration')

            # Dynamically calibrate color ranges based on the current frame
            red_hue_range = (red_min_hue, red_max_hue)
            blue_hue_range = (blue_min_hue, blue_max_hue)

            # Adjust camera exposure
            adjust_camera_exposure(cap, exposure_value=exposure_value)

            # Process the frame
            frame, center_red, center_blue, predicted_red, predicted_blue = process_frame(
                frame, kalman_red, kalman_blue, prev_center_red, prev_center_blue, red_hue_range, blue_hue_range)

            # Log frame data
            elapsed_time = time.time() - start_time
            print(f"Time: {elapsed_time:.2f}s | Red: {center_red} | Blue: {center_blue}")

            # Update previous centers for smoothing
            prev_center_red = center_red
            prev_center_blue = center_blue

            # Display the processed frame
            cv2.imshow("Object Tracking", frame)

            # Gracefully exit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Release the camera and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        print("Camera and windows have been released.")


# Start object tracking
if __name__ == "__main__":
    track_objects(camera_index=CAMERA_INDEX)

