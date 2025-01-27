import torch
import matplotlib.pyplot as plt
from .keypoint_extractor import keypoint_extractor
from .biomechanical_features import BiomechanicalFeatureExtractor
import numpy as np
from enum import Enum
from typing import Dict, Tuple


class PoseState(Enum):
    WAITING = "waiting"
    MOVEMENT = "movement"
    HOLD = "hold"

class YogaPoseStateMachine:
    def __init__(self, 
                 movement_threshold=0.5,
                 hold_threshold=0.2,
                 hold_duration=15,
                 noise_floor=0.1):
        
        self.state = PoseState.WAITING
        self.movement_threshold = movement_threshold
        self.hold_threshold = hold_threshold
        self.hold_duration = hold_duration
        self.noise_floor = noise_floor
        
        # State tracking variables
        self.frames_in_hold = 0
        self.movement_start_frame = None
        self.hold_start_frame = None
        self.current_frame = 0
        
        # Buffer for smoothing
        self.velocity_buffer = []
        self.buffer_size = 5
        self.velocity_whole_buffer = []
        
        # New: State history tracking
        self.state_history: Dict[int, Tuple[PoseState, int]] = {}
        self.last_state_change_frame = 0
        self.current_state_start_frame = 0
    
    def update_state_history(self, new_state: PoseState) -> None:
        """Update state history when state changes"""
        if new_state != self.state:
            # Calculate duration of the previous state
            duration = self.current_frame - self.current_state_start_frame
            
            # Store the state change with its duration
            self.state_history[self.current_state_start_frame] = (self.state, duration)
            
            # Update tracking variables
            self.last_state_change_frame = self.current_frame
            self.current_state_start_frame = self.current_frame
        
    def get_smoothed_velocity(self, velocity):
        """Smooth velocity using a simple moving average"""
        self.velocity_buffer.append(velocity)
        if len(self.velocity_buffer) > self.buffer_size:
            self.velocity_buffer.pop(0)
        self.velocity_whole_buffer.append(np.mean(self.velocity_buffer))
        return np.mean(self.velocity_buffer)
    
    def process_frame(self, squared_velocity):
        """Process a single frame of squared velocity data"""
        self.current_frame += 1
        smoothed_velocity = self.get_smoothed_velocity(squared_velocity)
        
        # Store previous state for comparison
        previous_state = self.state
        
        if self.state == PoseState.WAITING:
            if smoothed_velocity > self.movement_threshold:
                self.update_state_history(PoseState.MOVEMENT)
                self.state = PoseState.MOVEMENT
                self.movement_start_frame = self.current_frame
                print(f"Movement detected at frame {self.current_frame}")
                
        elif self.state == PoseState.MOVEMENT:
            if smoothed_velocity < self.hold_threshold:
                self.frames_in_hold += 1
                if self.frames_in_hold >= self.hold_duration:
                    self.update_state_history(PoseState.HOLD)
                    self.state = PoseState.HOLD
                    self.hold_start_frame = self.current_frame - self.hold_duration
                    print(f"Hold phase detected at frame {self.hold_start_frame}")
            else:
                self.frames_in_hold = 0
                
        elif self.state == PoseState.HOLD:
            if smoothed_velocity > self.movement_threshold:
                self.update_state_history(PoseState.MOVEMENT)
                self.state = PoseState.MOVEMENT
                self.frames_in_hold = 0
                print(f"Movement detected during hold at frame {self.current_frame}")
        
        return {
            'state': self.state,
            'movement_start': self.movement_start_frame,
            'hold_start': self.hold_start_frame,
            'current_frame': self.current_frame,
            'smoothed_velocity': smoothed_velocity,
            'state_history': self.state_history
        }
    
    def get_state_history(self) -> Dict[int, Tuple[PoseState, int]]:
        """Return the complete state history"""
        # Update the duration for the current state before returning
        current_duration = self.current_frame - self.current_state_start_frame
        history = self.state_history.copy()
        history[self.current_state_start_frame] = (self.state, current_duration)
        return history
    
    def reset(self):
        """Reset the state machine"""
        self.__init__(
            movement_threshold=self.movement_threshold,
            hold_threshold=self.hold_threshold,
            hold_duration=self.hold_duration,
            noise_floor=self.noise_floor
        )



def plot_velocity_data(velocity_buffer, raw_velocity_data):

    plt.figure(figsize=(10, 5))
    plt.plot(velocity_buffer, label='Smoothed Out Velocity')
    plt.plot(raw_velocity_data, alpha=0.5, label='Raw Velocity Data', color='orange')  # Lighter plot for noisy data
    plt.title('Velocity Whole Buffer')
    plt.xlabel('Frame')
    plt.ylabel('Velocity')
    plt.grid()
    plt.legend()
    plt.show()

# Call the function with the appropriate data

import subprocess
import os

def clip_video_with_ffmpeg(video_path, segment_info, output_dir):
    """
    Clips the video based on segment info using FFmpeg.

    Args:
        video_path (str): Path to the input video file.
        segment_info (dict): Dictionary containing start frame as key and (state, duration) as value.
        output_dir (str): Directory to save the clipped videos.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the video frame rate
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f"Error fetching frame rate: {result.stderr}")

    # Calculate frame rate (e.g., "30/1" -> 30.0)
    frame_rate_str = result.stdout.strip()
    num, denom = map(int, frame_rate_str.split("/"))
    frame_rate = num / denom

    for start_frame, (state, duration) in segment_info.items():
        # print("STATE", state.value)
        if state != PoseState.MOVEMENT:
            continue

        start_time = start_frame / frame_rate
        end_time = (start_frame + duration + int(frame_rate)) / frame_rate

        output_file = os.path.join(output_dir, f"{state.value}_{start_frame}_{start_frame + duration + int(frame_rate)}.mp4")

        # FFmpeg command to extract the segment
        cmd = [
            "ffmpeg", "-y",  # Overwrite output file if it exists
            "-i", video_path,
            "-ss", f"{start_time:.2f}",
            "-to", f"{end_time:.2f}",
            "-c:v", "libx264",  # Video codec
            "-preset", "fast",  # Encoding speed
            "-crf", "23",  # Constant Rate Factor (quality)
            output_file
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error clipping video for segment {start_frame}: {result.stderr}")
        else:
            print(f"Segment saved: {output_file}")







def segment_video(video_path, 
                         movement_threshold=0.3, 
                         hold_threshold=0.1, 
                         hold_duration=30, return_feats=False):
    """
    Analyze video states and return state history
    
    Args:
        video_path (str): Path to input video
        movement_threshold (float, optional): Threshold for detecting movement. Defaults to 0.3.
        hold_threshold (float, optional): Threshold for detecting hold. Defaults to 0.1.
        hold_duration (int, optional): Duration to consider a hold state. Defaults to 30.
    
    Returns:
        dict: State history mapping start frames to (state, duration)
    """
    extractor = BiomechanicalFeatureExtractor()
    
    
    #  MEDIAPIPE KEY POINT
    features = keypoint_extractor(video_path)
    # RETURN np.ndarray( TOTAL_NUMBER_OF_FRAMES_IN_VIDEO, 33(JOINTS), 3(x,y,z)) ,i.e. (Number_frames, 33, 3) in numpy array
    features = torch.from_numpy(features)
    
    velocity = extractor.extract_features(features)["Joint Acceleration"]

    v = torch.sqrt(velocity[..., 0]**2 + velocity[...,1]**2 + velocity[...,2]**2)
    velocity_magnitude = v.sum(dim=-1) 
    velocity_magnitude = velocity_magnitude.clamp_(min=0, max=1.5).pow_(2).clamp_(max=1.5)

    state_machine = YogaPoseStateMachine(
        movement_threshold=movement_threshold,
        hold_threshold=hold_threshold,
        hold_duration=hold_duration 
    )

    for frame_velocity in velocity_magnitude:
        state_machine.process_frame(frame_velocity)

    if return_feats:
        return (state_machine.get_state_history(), features, velocity_magnitude)
    else:
        return state_machine.get_state_history()




if __name__=="__main__":
    print(segment_video('tiktok_data/mountain/first.mp4'))
    
