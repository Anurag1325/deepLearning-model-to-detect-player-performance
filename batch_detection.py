import cv2
import numpy as np
from ultralytics import YOLO

# Path to the YOLO model and input video
model_path = r"models\best.pt"
video_path = r"input_videos\08fd33_4.mp4"

# Load YOLO model
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

player_counter = 1  # Player number counter for labeling
batch_size = 8  # Define your batch size for processing frames

# Camera movement estimation setup (same as before)
prev_gray = None
prev_points = None
new_points = None  # Define new_points to avoid the error

def estimate_camera_movement(prev_gray, frame_gray):
    points = cv2.goodFeaturesToTrack(frame_gray, mask=None, **corner_params)
    
    if prev_points is not None:
        new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None)
        if new_points is not None:
            movement = np.mean(new_points - prev_points, axis=0)
            # Flatten movement to ensure it is a 1D array
            movement = np.array(movement).flatten()[:2]  # Ensure it's always 2D (x, y)
            return movement, new_points
    return np.array([0, 0]), points  # Return a 2D array (x, y)

corner_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

frames_batch = []  # Initialize a list to store frames for batch processing
gray_frames = []  # Define gray_frames outside the loop to prevent NameError

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Collect frames for the batch
    frames_batch.append(frame)
    
    # If we've reached the batch size, process the batch
    if len(frames_batch) == batch_size:
        # Convert all frames in the batch to grayscale for optical flow calculation
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_batch]

        # Estimate camera movement for each frame in the batch
        for idx, gray_frame in enumerate(gray_frames):
            if prev_gray is not None:
                movement, new_points = estimate_camera_movement(prev_gray, gray_frame)
                # Now movement should be a 1D array with 2 elements (x and y)
                if len(movement) != 2:
                    print("Error: Invalid movement vector.")
                    continue
                # Apply transformation based on movement
                M = np.float32([[1, 0, movement[0]], [0, 1, movement[1]]])  # Translation matrix
                frames_batch[idx] = cv2.warpAffine(frames_batch[idx], M, (frames_batch[idx].shape[1], frames_batch[idx].shape[0]))

        # Run YOLOv8 detection on the entire batch of frames
        results = model(frames_batch)  # Process the batch of frames

        # Loop through each frame and its corresponding results
        for idx, frame in enumerate(frames_batch):
            boxes = results[idx].boxes

            # Loop through each detection and draw bounding boxes and labels
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls_id = int(box.cls[0])
                cls_name = results[idx].names[cls_id]

                # Draw bounding box
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add label with player number and confidence score
                label = f"Player {player_counter}: {cls_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Increment player counter for each player detected
                player_counter += 1

            # Show the frame with detections and player numbers
            cv2.imshow("Detections with Player Numbers and Camera Movement", frame)

        # Clear the frames batch after processing
        frames_batch.clear()

    # Update previous frame for optical flow
    if gray_frames:  # Only update if gray_frames exists
        prev_gray = gray_frames[-1]  # Update previous frame with the last frame in the batch
        prev_points = new_points if new_points is not None else prev_points

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
