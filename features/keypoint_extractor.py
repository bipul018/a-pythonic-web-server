import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
import time

class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process a single frame and return the visualization and 3D points."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Create visualization
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # Extract 3D points if available
        if results.pose_world_landmarks:
            # Convert landmarks to numpy array
            points_3d = np.array([[lm.x, lm.y, lm.z] 
                                for lm in results.pose_world_landmarks.landmark])
            return annotated_frame, points_3d
        
        return annotated_frame, None

    def process_video(self, input_path: str, output_path: str) -> Optional[np.ndarray]:
        """
        Process entire video and return array of 3D points.
        Returns array of shape (num_frames, 33, 3) if successful.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Storage for 3D points
        all_points = []
        
        # Process frame by frame
        frame_count = 0
        start_time = time.time()
        # skip = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            annotated_frame, points_3d = self.process_frame(frame)
            
            # Write visualization
            out.write(annotated_frame)
            
            # Store points if detected
            if points_3d is not None:
                all_points.append(points_3d)
            else:
                print(f"Warning: No pose detected in frame {frame_count}")
                # Fill with zeros to maintain array shape
                # all_points.append(np.zeros((33, 3)))

            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({fps:.2f} fps)")
        
        # Release resources
        cap.release()
        out.release()
        
        if len(all_points) > 0:
            return np.array(all_points)  # Shape: (num_frames, 33, 3)
        return None

def keypoint_extractor(input_file = None):
    # Example usage
    if input_file is None:
        input_file = "tiktok_data/first.mp4"
    extractor = PoseExtractor()
    try:
        points_3d = extractor.process_video(
            input_path=input_file,
            output_path="output_video.mp4"
        )
        if points_3d is not None:
            print(f"Extracted 3D points shape: {points_3d.shape}")
            # Save points to file
            # np.save("pose_points_3d.npy", points_3d)
            return points_3d
        else:
            print("No poses detected in video")
            return None
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__=="__main__":
    keypoint_extractor()
