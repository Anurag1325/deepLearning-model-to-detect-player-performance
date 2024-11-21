import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("models/best.pt")  # Path to your trained YOLOv8 model

# Video path
video_path = "input_videos/08fd33_4.mp4"

def process_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the frame
        results = model(frame)  # Detect objects in the frame
        boxes = []
        confidences = []
        class_ids = []

        # Extract bounding boxes, class IDs, and confidence scores
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
                conf = box.conf[0]  # Get the confidence score
                class_id = int(box.cls[0])  # Get the class ID
                
                if class_id == 0:  # Only consider players (class ID = 0)
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])  # Convert to integer values
                    confidences.append(float(conf))  # Convert to float
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        # Ensure `boxes` is a numpy array and `confidences` is a list of floats
        boxes = np.array(boxes)  # Convert boxes to a numpy array for NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences, score_threshold=0.3, nms_threshold=0.4)

        if len(indices) > 0:
            for i in indices.flatten():  # Flatten the indices array
                x1, y1, x2, y2 = boxes[i]  # Get the box coordinates
                label = 'Player'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the processed frame
        cv2.imshow("Frame", frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process the video
process_video(video_path)
