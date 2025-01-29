import asyncio
import io
import av
import numpy
import cv2
import subprocess
import time
import os
import tempfile
import pathlib
import fractions

import math


def map_to_range(input_value, n, m):
    """Maps input_value from [0, n-1] to [0, m-1] using rounding."""
    return int(round((input_value / (n - 1)) * (m - 1)))  #Scale 

def find_first_match(data, field, value):
    """Finds the first dictionary in data where field == value."""
    for item in data:
        if item.get(field) == value:  # Use get() to handle missing keys gracefully
            return item
    return None  # Return None if no match is found


class FrameGenStreamAV:
    def __init__(self, file_or_obj,
                 fix_fps = None, fix_frames = None):
        self.in_container = av.open(file_or_obj, mode='r')
        vid_stream = self.in_container.streams.video[0]

        vid_stream.thread_type = "AUTO" # makes it go faster
        self.shape = (int(vid_stream.height), int(vid_stream.width), 3)
        self.max_frames = int(vid_stream.frames)
        print(f"The vid stream object is {vid_stream}, shape = {self.shape}, max_frames = {self.max_frames}, file obj is of {len(file_or_obj.getvalue())}")
        self.duration = float(vid_stream.duration * vid_stream.time_base)
        self.fps = self.max_frames/self.duration
        self.total_size = self.shape[0] * self.shape[1] * self.shape[2]
        # calculate the desired fps to capture at/interpolate at
        self._frame_genr = self.in_container.decode(video=0)
        if fix_frames is not None:
            self.desired_frames = fix_frames
        elif fix_fps is not None:
            self.desired_frames = math.floor(fix_fps * self.duration)
        else:
            self.desired_frames = self.max_frames
        if self.desired_frames > self.max_frames:
            raise Exception(f"Desired frames is {self.desired_frames} which is greater than total frames that is {self.max_frames}")
        self.finx = -1
        self.actual_frame = 0
    def __enter__(self):
        return self
    def close(self):
        self.in_container.close()
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def all_frames(self):
        """Returns the entire video frames remaining to decode. Also will close `self` on return."""
        frames=None
        while (frame:=self.next_frame()) is not None:
            frame=frame.reshape((1,)+frame.shape)
            if frames is None:
                frames=frame
            else:
                frames=numpy.concatenate((frames,frame))
        self.close()
        return frames
    def next_frame(self):
        if self.actual_frame >= self.max_frames:
            return None
        old_finx = self.finx
        in_frame = None
        while old_finx == self.finx:
            in_frame = None
            try:
                in_frame = next(self._frame_genr)
            except StopIteration:
                in_frame = None
                break
            self.actual_frame += 1
            self.finx = map_to_range(self.actual_frame,
                                     self.max_frames,
                                     self.desired_frames)
        if in_frame is not None:
            return in_frame.to_ndarray(format='rgb24')
        return None
    def ts_ms(self):
        return 1000 * (self.actual_frame / self.fps)

class FrameGenStreamCV:
    def __init__(self, file_or_obj,
                 fix_fps = None, fix_frames = None):
        self.in_container = av.open(file_or_obj, mode='r')
        vid_stream = self.in_container.streams.video[0]

        vid_stream.thread_type = "AUTO" # makes it go faster
        self.shape = (int(vid_stream.height), int(vid_stream.width), 3)
        self.max_frames = int(vid_stream.frames)
        print(f"The vid stream object is {vid_stream}, shape = {self.shape}, max_frames = {self.max_frames}, file obj is of {len(file_or_obj.getvalue())}")
        self.duration = float(vid_stream.duration * vid_stream.time_base)
        self.fps = self.max_frames/self.duration
        self.total_size = self.shape[0] * self.shape[1] * self.shape[2]
        # calculate the desired fps to capture at/interpolate at
        self._frame_genr = self.in_container.decode(video=0)
        if fix_frames is not None:
            self.desired_frames = fix_frames
        elif fix_fps is not None:
            self.desired_frames = math.floor(fix_fps * self.duration)
        else:
            self.desired_frames = self.max_frames
        if self.desired_frames > self.max_frames:
            raise Exception(f"Desired frames is {self.desired_frames} which is greater than total frames that is {self.max_frames}")
        self.finx = -1
        self.actual_frame = 0
    def __enter__(self):
        return self
    def close(self):
        self.in_container.close()
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def all_frames(self):
        """Returns the entire video frames remaining to decode. Also will close `self` on return."""
        frames=None
        while (frame:=self.next_frame()) is not None:
            frame=frame.reshape((1,)+frame.shape)
            if frames is None:
                frames=frame
            else:
                frames=numpy.concatenate((frames,frame))
        self.close()
        return frames
    def next_frame(self):
        if self.actual_frame >= self.max_frames:
            return None
        old_finx = self.finx
        in_frame = None
        while old_finx == self.finx:
            in_frame = None
            try:
                in_frame = next(self._frame_genr)
            except StopIteration:
                in_frame = None
                break
            self.actual_frame += 1
            self.finx = map_to_range(self.actual_frame,
                                     self.max_frames,
                                     self.desired_frames)
        if in_frame is not None:
            return in_frame.to_ndarray(format='rgb24')
        return None
    def ts_ms(self):
        return 1000 * (self.actual_frame / self.fps)
