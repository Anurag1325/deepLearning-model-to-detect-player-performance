import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# Paths
video_path = "input_videos/08fd33_4.mp4"  # Replace with your video path
model_path = "models/best.pt"  # Replace with your model path

# Constants
FPS = 30  # Frames per second of the video
SKIP_FRAMES = 10  # Process every 10th frame for efficiency
DISTANCE_THRESHOLD = 50  # Threshold for ball possession distance (in pixels)
MIN_DETECTION_AREA = 100  # Minimum area in pixels for a valid detection

# Initialize Data structures
metrics_data = []
frame_count = 0
predicted_labels = []
true_labels = []

# Load YOLO model
model = YOLO(model_path)

def get_detections(frame, model):
    """Get detections from YOLO model."""
    results = model(frame)
    detections = []
    
    for result in results:
        if hasattr(result, 'boxes'):
            boxes = result.boxes.xyxy  # Get the xyxy boxes
            confidences = result.boxes.conf  # Get confidence scores
            for idx, box in enumerate(boxes):
                confidence = confidences[idx].item()
                if confidence >= 0.2:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    class_name = result.names[result.boxes.cls[idx].item()]  # Get class name based on idx
                    detections.append({
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': confidence,
                        'class': class_name  # Updated to reflect class name directly
                    })
    return detections

def process_video(video_path, model):
    """Process video to calculate metrics and show results."""
    cap = cv2.VideoCapture(video_path)
    global frame_count, predicted_labels, true_labels

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % SKIP_FRAMES == 0:
            # Get detections for the current frame
            detections = get_detections(frame, model)
            current_positions = {}
            detected_classes = []

            # Process each detection (player or ball)
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_id = detection['class']
                detected_classes.append(class_id)

                # Here, assume we have the true label from ground truth for the frame
                true_class = 'person'  # Replace with ground truth class for the frame
                true_labels.append(true_class)  # Append the correct true label

            # If no detections were made, append 'no_detection' for that frame
            if not detected_classes:
                predicted_labels.append(['no_detection'])  # No detection for this frame
            else:
                predicted_labels.append(detected_classes)  # Add detected classes

            # Draw bounding boxes and metrics on the frame
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_id = detection['class']
                if class_id == 'person':  # Player
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                elif class_id == 'sports ball':  # Ball
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Display the frame with metrics
            cv2.imshow("Football Match - Player Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()

    # After processing all frames, calculate the performance metrics
    calculate_performance_metrics()

def calculate_performance_metrics():
    """Calculate and visualize the performance metrics."""
    # Ensure the lengths of true_labels and predicted_labels match
    if len(true_labels) != len(predicted_labels):
        print(f"Mismatch: true_labels length: {len(true_labels)} vs predicted_labels length: {len(predicted_labels)}")
        return

    # Flatten lists of true and predicted labels
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    predicted_labels_flat = [item for sublist in predicted_labels for item in sublist]

    # Debug: Print lengths of flattened lists
    print(f"True labels: {len(true_labels_flat)}")
    print(f"Predicted labels: {len(predicted_labels_flat)}")

    # Calculate F1 Score (Multiclass version)
    try:
        f1 = f1_score(true_labels_flat, predicted_labels_flat, average='weighted')

        # Confusion Matrix
        cm = confusion_matrix(true_labels_flat, predicted_labels_flat)

        # Visualize Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['person', 'sports ball'], yticklabels=['person', 'sports ball'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Display F1 Score
        print(f"F1 Score (Weighted): {f1:.2f}")

    except Exception as e:
        print(f"Error calculating performance metrics: {e}")

# Main execution
def main():
    # Load the YOLOv8 model
    yolo_model = YOLO(model_path)
    
    # Process the video and calculate metrics
    process_video(video_path, yolo_model)

if __name__ == '__main__':
    main()
