from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time

# Function to calculate IoU
def calculate_iou(pred_box, gt_boxes):
    max_iou = 0
    for gt_box in gt_boxes:
        x1, y1, x2, y2 = map(int, gt_box)
        pred_x1, pred_y1, pred_x2, pred_y2 = map(int, pred_box.xyxy[0])

        # Calculate intersection coordinates
        inter_x1 = max(x1, pred_x1)
        inter_y1 = max(y1, pred_y1)
        inter_x2 = min(x2, pred_x2)
        inter_y2 = min(y2, pred_y2)

        # Calculate intersection area
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union area
        gt_area = (x2 - x1) * (y2 - y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        union_area = gt_area + pred_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        max_iou = max(max_iou, iou)

    return max_iou

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path to the YOLO model and input video
model_path = r"models\best.pt"
video_path = r"input_videos\08fd33_4.mp4"

# Load YOLO model
model = YOLO(model_path)  # Initialize YOLO model
model.to(device)  # Move model to GPU if available

cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Initialize heatmap as a black canvas (same size as the frame)
heatmap = None
alpha = 0.7  # Opacity for heatmap overlay
total_detections = 0  # Count of all detections
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
frame_counter = 0
n = 2  # Process every 2nd frame
min_confidence = 0.5  # Adjust this threshold
ground_truth_boxes = []  # List of ground truth bounding boxes for each frame (if available)

start_time = time.time()  # Start the timer

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % n != 0:
        continue  # Skip frames

    # Resize frame to speed up processing (optional)
    frame_resized = cv2.resize(frame, (640, 360))

    # Run YOLOv8 detection
    results = model(frame_resized)  # Use the model to make predictions

    # Check if results are valid
    if results is None or not results:
        continue  # Skip if no results found for this frame

    # Filter results based on confidence threshold
    boxes = results[0].boxes
    boxes = boxes[boxes.conf > min_confidence]  # Filter by confidence

    # If heatmap is not initialized, create it with the same size as the frame
    if heatmap is None:
        heatmap = np.zeros_like(frame, dtype=np.float32)

    # Loop through each detection and accumulate positions in the heatmap
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calculate the center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Increase the intensity at the center of the bounding box
        heatmap[center_y, center_x] += 1
        total_detections += 1  # Count each detection

        # Compute IoU with ground truth for calculating TP, FP, FN
        # Assuming ground_truth_boxes is available, otherwise skip or simulate GT
        iou = calculate_iou(box, ground_truth_boxes)
        if iou > 0.5:  # IoU threshold to consider it a True Positive
            total_true_positives += 1
        else:
            total_false_positives += 1

    # Calculate false negatives (GT objects not detected)
    total_false_negatives = len(ground_truth_boxes) - total_true_positives

    # Normalize the heatmap to scale between 0 and 255
    heatmap_display = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_display = np.uint8(heatmap_display)

    # Apply color map to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original frame
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

    # Display the results on the frame (Optional)
    cv2.putText(overlay, f"Total Detections: {total_detections}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with heatmap overlay
    cv2.imshow("Heatmap Overlay", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate final statistics
precision = total_true_positives / (total_true_positives + total_false_positives)
recall = total_true_positives / (total_true_positives + total_false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
average_activity_per_frame = total_detections / cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(f"Total Detections: {total_detections}")
print(f"True Positives: {total_true_positives}")
print(f"False Positives: {total_false_positives}")
print(f"False Negatives: {total_false_negatives}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Average Detections per Frame: {average_activity_per_frame:.2f}")
print(f"Average FPS: {1 / (end_time - start_time):.2f}")

end_time = time.time()  # End the timer
print(f"Total processing time: {end_time - start_time:.2f} seconds")

cap.release()
cv2.destroyAllWindows()
