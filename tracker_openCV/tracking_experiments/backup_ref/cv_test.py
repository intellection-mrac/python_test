import cv2
import yt_dlp
import matplotlib.pyplot as plt

def get_video_url(youtube_url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict['formats'][0]['url']
        return video_url

# Get YouTube URL from the user
youtube_url = input("Enter the YouTube video URL: ")

# Extract the video stream URL using yt-dlp
video_url = get_video_url(youtube_url)

# Open the video stream using OpenCV
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit()

frame_count = 0

# Set up matplotlib for real-time display
plt.ion()
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert BGR frame to RGB for display with matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    ax.imshow(frame_rgb)
    ax.axis('off')  # Hide axes
    plt.draw()
    plt.pause(0.01)  # Pause to update the plot

    frame_count += 1

    # Optionally, print the frame count
    print(f"Processed frame {frame_count}")

# Release resources
cap.release()
cv2.destroyAllWindows()

