import cv2
import os
import pandas as pd
from ultralytics import YOLO
from team_assigner.team_assigner import TeamAssigner
from utils import read_video, save_video  # Assuming these are utility functions in your project
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

# Define field boundaries (example for a 1920x1080 video)
FIELD_X_MIN = 0  # minimum x-coordinate (left)
FIELD_X_MAX = 1920  # maximum x-coordinate (right)
FIELD_Y_MIN = 0  # minimum y-coordinate (top)
FIELD_Y_MAX = 1180  # maximum y-coordinate (bottom)
CONFIDENCE_THRESHOLD = 0.2  # Confidence threshold to filter out low confidence detections
MIN_DETECTION_AREA = 100  # Minimum area in pixels for a valid detection

# Function to extract detections from the YOLO model
def get_detections(frame, model):
    results = model(frame)
    detections = []

    if isinstance(results, list):
        for result in results:
            if hasattr(result, 'boxes'):
                boxes = result.boxes.xyxy  # Get the xyxy boxes
                confidences = result.boxes.conf  # Get confidence scores
                for idx, box in enumerate(boxes):
                    confidence = confidences[idx].item()
                    if confidence >= CONFIDENCE_THRESHOLD:  # Filter detections based on confidence
                        x1, y1, x2, y2 = box
                        detections.append({
                            'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                            'confidence': confidence,
                            'class': result.names[0]
                        })
    else:
        if hasattr(results, 'boxes'):
            boxes = results.boxes.xyxy
            confidences = results.boxes.conf
            for idx, box in enumerate(boxes):
                confidence = confidences[idx].item()
                if confidence >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box
                    detections.append({
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': confidence,
                        'class': results.names[0]
                    })
    
    return detections

# Filter out detections outside the field boundaries
def filter_field_detections(detections):
    filtered_detections = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        # Filter out detections outside the defined field
        if x1 >= FIELD_X_MIN and x2 <= FIELD_X_MAX and y1 >= FIELD_Y_MIN and y2 <= FIELD_Y_MAX:
            filtered_detections.append(detection)
    return filtered_detections

# Filter out detections that are too small
def filter_small_detections(detections):
    filtered_detections = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        area = (x2 - x1) * (y2 - y1)
        if area >= MIN_DETECTION_AREA:
            filtered_detections.append(detection)
    return filtered_detections

# Main function to process the video, assign teams, and save results
def process_video(video_path, model, team_assigner, output_dir='output_videos'):
    # Read the video frames
    video_frames = read_video(video_path)
    if not video_frames:
        print("Error: No frames were loaded from the video.")
        return

    # Initialize DataFrame to store detection results
    metrics_data = []
    output_video_frames = []

    # Loop through frames for detection, team assignment, and camera movement
    for frame_idx, frame in enumerate(video_frames):
        # Get detections for the current frame (using YOLO model)
        detections = get_detections(frame, model)

        # Filter detections that are outside the field and too small
        detections = filter_field_detections(detections)
        detections = filter_small_detections(detections)

        # Only process frames with more than 1 detection (needed for team assignment)
        if len(detections) > 1:
            # Assign teams based on position and color using KMeans
            team_assignments = team_assigner.assign_teams(frame, detections)
        else:
            # If only 1 detection, assign it to a default team or skip team assignment
            team_assignments = [1 for _ in detections]  # Assign a default team

        # Save metrics data (frame index, team, bounding box, etc.)
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            team = team_assignments[idx]
            confidence = detection['confidence']
            metrics_data.append({
                'frame': frame_idx,
                'team': team,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class': detection['class']
            })
        
        # Draw bounding boxes and team assignments on the frame
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            team = team_assignments[idx]
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Display team, bounding box, and confidence on the frame
            cv2.putText(frame, f"Team {team}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"BBox: [{int(x1)}, {int(y1)}] [{int(x2)}, {int(y2)}]", 
                        (int(x1), int(y1)-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame with overlaid metrics
        cv2.imshow("Football Match - Player Detection", frame)

        # Check if the user pressed the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add the processed frame to the output video list
        output_video_frames.append(frame)

    # Save the processed frames to an output video file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_video_path = os.path.join(output_dir, 'output_video.avi')
    save_video(output_video_frames, output_video_path)
    print(f"Processed video saved to {output_video_path}")

    # Save metrics data to DataFrame and CSV
    df = pd.DataFrame(metrics_data)
    metrics_output_path = os.path.join(output_dir, 'detection_metrics.csv')
    df.to_csv(metrics_output_path, index=False)
    print(f"Detection metrics saved to {metrics_output_path}")

# Main execution
def main():
    # Load the YOLOv8 model
    model_path = 'models/best.pt'
    yolo_model = YOLO(model_path)

    # Initialize TeamAssigner (assumes it's a class that assigns teams based on player detections)
    team_assigner = TeamAssigner()

    # Path to input video
    video_path = 'input_videos/08fd33_4.mp4'

    # Process the video
    process_video(video_path, yolo_model, team_assigner)

if __name__ == '__main__':
    main()
