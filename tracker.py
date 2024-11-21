from ultralytics import YOLO
from deep_sort_realtime import DeepSort
import torch
import numpy as np

class Tracker:
    def __init__(self, model_path):
        # Initialize YOLO model and DeepSORT tracker
        self.model = YOLO(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.tracker = DeepSort(max_age=30, nn_budget=70)  # Adjust parameters as needed

    def detect_frames(self, frames):
        """
        Perform batch detection on video frames using the YOLO model.
        Returns detected boxes, confidences, and class IDs for each frame.
        """
        batch_size = 4
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            if len(batch_frames) == 0:
                continue
            detections_batch = self.model(batch_frames, conf=0.3)  # Adjust confidence threshold
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames):
        """
        Perform object tracking across frames using YOLO detections and DeepSORT.
        Returns track data (track ID, bounding box) for each frame.
        """
        # Get detections for all frames
        detections = self.detect_frames(frames)

        all_tracks = []
        for frame_num, detection in enumerate(detections):
            try:
                # Extract bounding boxes (xyxy), confidences, and class IDs
                boxes = detection.boxes.xyxy.cpu().numpy()  # Convert to NumPy arrays
                confidences = detection.boxes.conf.cpu().numpy()
                class_ids = detection.boxes.cls.cpu().numpy()

            except AttributeError:
                # If there are no detections in the frame
                print(f"No detections for frame {frame_num}")
                continue

            # Use DeepSORT to update object tracks
            tracks = self.tracker.update_tracks(boxes, confidences, class_ids, frame_num=frame_num)

            # Collect the confirmed tracks
            frame_tracks = [
                {"track_id": track.track_id, "bbox": track.to_ltwh()}  # Get bounding box in [left, top, width, height]
                for track in tracks if track.is_confirmed()  # Only add confirmed tracks
            ]
            all_tracks.append(frame_tracks)

        return all_tracks
