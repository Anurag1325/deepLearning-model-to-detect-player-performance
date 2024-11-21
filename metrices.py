import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
import os

# Paths
video_path = "input_videos/08fd33_4.mp4"  # Replace with your video path
model_path = "models/best.pt"  # Replace with your model path

# Constants
FPS = 30  # Frames per second of the video
SKIP_FRAMES = 10  # Process every 10th frame for efficiency
DISTANCE_THRESHOLD = 50  # Threshold for ball possession distance (in pixels)
MIN_DETECTION_AREA = 100  # Minimum area in pixels for a valid detection

# Initialize Data structures
player_speed = defaultdict(list)  # Speed of each player
player_acceleration = defaultdict(list)  # Acceleration of each player
ball_interactions = defaultdict(int)  # Ball interactions per player
possession_time = defaultdict(float)  # Possession time per player
previous_positions = {}  # Stores previous positions of players and ball
previous_speeds = defaultdict(float)  # Stores previous speeds of players

# Load YOLO model
model = YOLO(model_path)

def calculate_speed(pos1, pos2, time):
    """Calculate speed given two positions and time."""
    distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
    return distance / time

def calculate_acceleration(speed1, speed2, time):
    """Calculate acceleration given two speeds and time."""
    return (speed2 - speed1) / time

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
    frame_count = 0
    metrics_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % SKIP_FRAMES == 0:
            # Get detections for the current frame
            detections = get_detections(frame, model)
            current_positions = {}

            # Process each detection (player or ball)
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                class_id = detection['class']

                if class_id == 'person':  # Player detection
                    player_id = f"player_{frame_count}_{cx}_{cy}"
                    current_positions[player_id] = (cx, cy)

                    # Speed and acceleration calculations
                    if player_id in previous_positions:
                        prev_pos = previous_positions[player_id]
                        speed = calculate_speed(prev_pos, (cx, cy), SKIP_FRAMES / FPS)
                        player_speed[player_id].append(speed)

                        # Calculate acceleration
                        if len(player_speed[player_id]) > 1:
                            acceleration = calculate_acceleration(
                                player_speed[player_id][-2], player_speed[player_id][-1], SKIP_FRAMES / FPS
                            )
                            player_acceleration[player_id].append(acceleration)

                    previous_positions[player_id] = (cx, cy)
                
                elif class_id == 'sports ball':  # Ball detection
                    ball_pos = (cx, cy)

                    # Ball interaction and possession time calculation
                    for player_id, pos in current_positions.items():
                        if np.linalg.norm(np.array(pos) - np.array(ball_pos)) < DISTANCE_THRESHOLD:
                            ball_interactions[player_id] += 1
                            possession_time[player_id] += SKIP_FRAMES / FPS

            # Add metrics data for this frame
            for player_id in current_positions:
                avg_speed = np.mean(player_speed.get(player_id, [0]))  # Ensure default if no speed
                avg_acceleration = np.mean(player_acceleration.get(player_id, [0]))  # Ensure default if no acceleration
                interactions = ball_interactions.get(player_id, 0)
                poss_time = possession_time.get(player_id, 0)
                
                metrics_data.append({
                    'frame': frame_count,
                    'player_id': player_id,
                    'avg_speed': avg_speed,
                    'avg_acceleration': avg_acceleration,
                    'ball_interactions': interactions,
                    'possession_time': poss_time
                })

            # Debugging: Print out the metrics data collected for each frame
            print(f"Collected Metrics for Frame {frame_count}:")
            for data in metrics_data[-len(current_positions):]:  # Print only the latest player data
                print(data)

            # Draw bounding boxes and metrics on the frame
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_id = detection['class']
                if class_id == 'person':  # Player
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Display player speed and acceleration
                    player_id = f"player_{frame_count}_{(x1 + x2) / 2}_{(y1 + y2) / 2}"
                    speed = np.mean(player_speed.get(player_id, [0]))
                    acceleration = np.mean(player_acceleration.get(player_id, [0]))
                    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Accel: {acceleration:.2f} m/s^2", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                elif class_id == 'sports ball':  # Ball
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Display the frame with metrics
            cv2.imshow("Football Match - Player Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()

    # Save metrics to DataFrame
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv("metrics_result.csv", index=False)  # Saving to CSV
        print("Metrics saved to 'metrics_result.csv'")
    else:
        print("No metrics data collected.")

    # Display the metrics DataFrame
    if metrics_data:
        print(metrics_df)

# Main execution
def main():
    # Load the YOLOv8 model
    yolo_model = YOLO(model_path)
    
    # Process the video and calculate metrics
    process_video(video_path, yolo_model)

if __name__ == '__main__':
    main()
