import cv2
from ultralytics import YOLO
import numpy as np

# Path to the YOLO model and input video
model_path = r"models\best.pt"
video_path = r"input_videos\08fd33_4.mp4"

# Load YOLO model
model = YOLO(model_path)

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Resize frame for faster processing and avoid large resolution issues
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

player_counter = 1  # Player number counter for labeling

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)
    boxes = results[0].boxes

    # Loop through each detection and draw bounding boxes and labels
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]

        # Draw bounding box
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add label with player number and confidence score
        label = f"Player {player_counter}: {cls_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Increment player counter for each player detected
        player_counter += 1

    # Show the frame with detections and player numbers
    cv2.imshow("Detections with Player Numbers", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
