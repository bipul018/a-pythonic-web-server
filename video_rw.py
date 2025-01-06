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
import draw_landmarks

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

class VideoFromFrameAV:
    """
    When a desired output sink is provided, writes the video data into that sink,
    else will create a buffer and write to that buffer as bufferio
    """
    def __init__(self, width, height, new_fps, output_file_or_obj = None):
        if output_file_or_obj is None:
            self.dst_obj = io.BytesIO()
            self._was_temp = True
            output_format = 'mp4'
        else:
            self.dst_obj = output_file_or_obj
            self._was_temp = False
            try:
                output_file_or_obj.name
                output_format = None
            except:
                output_format = 'mp4'
            #TODO:: FIX THIS , THE CODE LATER USES self.file_name
            #self.file_name = out_name
        self.shape = (height, width, 3)
        self.out_container = av.open(self.dst_obj, mode='w', format=output_format)
        frac_fps = fractions.Fraction(new_fps).limit_denominator(10000)
        self.out_stream = self.out_container.add_stream('mpeg4', rate=frac_fps)
        self.out_stream.width = width
        self.out_stream.height = height
        self.finx = 0
        self.fps = new_fps
        self._terminated = False
    def __enter__(self):
        return self
    def bytes(self):
        """Returns the contents of video buffer. Only allowed to call if terminated already"""
        if not self._terminated:
            raise Exception("Tried to get video bytes before completing encoding")
        if isinstance(self.dst_obj, io.BytesIO):
            current_pos = self.dst_obj.tell()
            self.dst_obj.seek(0)
            data = self.dst_obj.read()
            self.dst_obj.seek(current_pos)
            return data
        else:
            with open(self.file_name, 'rb') as f:
                return f.read()
    def terminate(self):
        if not self._terminated:
            self._terminated = True
            # Flush stream
            for packet in self.out_stream.encode():
                self.out_container.mux(packet)
            # Close the file
            self.out_container.close()
    def close(self):
        self.terminate()
        if self._was_temp:
            self.dst_obj.close()
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def write_frame(self, np_frame):
        assert (np_frame.shape[0] == self.shape[0]) and (np_frame.shape[1] == self.shape[1]) and (np_frame.shape[2] == self.shape[2])
        frame = av.VideoFrame.from_ndarray(np_frame, format="rgb24")
        assert (frame.height == self.shape[0]) and (frame.width == self.shape[1])
        print(f"The frame type and shape is {type(frame)}, {frame.shape}")
        for packet in self.out_stream.encode(frame):
            self.out_container.mux(packet)
        self.finx+=1
    def ts_ms(self):
        return 1000 * (self.finx/self.fps)

class VideoFromFrameCV:
    """
    When a desired output sink is provided, writes the video data into that sink,
    else will create a buffer and write to that buffer as bufferio
    """
    def __init__(self, width, height, new_fps, output_file_or_obj = None):
        if output_file_or_obj is None:
            self.tempfile = tempfile.NamedTemporaryFile(mode='wb', suffix = ".mp4")
            # self.dst_obj = io.BytesIO()
            self.dst_obj = self.tempfile.name
            self._was_temp = True
            output_format = 'mp4'
        else:
            self.dst_obj = output_file_or_obj
            self._was_temp = False
            try:
                output_file_or_obj.name
                output_format = None
            except:
                #TODO:: Wont support writing to user given bytes io for now
                raise Exception("Opencv version doesnot support writing to user given bytes io for now")
                output_format = 'mp4'
            #self.file_name = out_name
        self.shape = (height, width, 3)
        frac_fps = fractions.Fraction(new_fps).limit_denominator(10000)

        #TODO:: Support detection of mutiple codecs
        #self.cv_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.cv_fourcc = cv2.VideoWriter_fourcc(*"h264")
        self.cv_out = cv2.VideoWriter(self.dst_obj, self.cv_fourcc, float(frac_fps), (width, height))
        #self.out_container = av.open(self.dst_obj, mode='w', format=output_format)
        #self.out_stream = self.out_container.add_stream('mpeg4', rate=frac_fps)
        #self.out_stream.width = width
        #self.out_stream.height = height

        self.finx = 0
        self.fps = new_fps
        self._terminated = False
    def __enter__(self):
        return self
    def bytes(self):
        """Returns the contents of video buffer. Only allowed to call if terminated already"""
        if not self._terminated:
            raise Exception("Tried to get video bytes before completing encoding")
        if isinstance(self.dst_obj, io.BytesIO):
            current_pos = self.dst_obj.tell()
            self.dst_obj.seek(0)
            data = self.dst_obj.read()
            self.dst_obj.seek(current_pos)
            return data
        else:
            #with open(self.file_name, 'rb') as f:
            with open(self.dst_obj, 'rb') as f:
                return f.read()
    def terminate(self):
        if not self._terminated:
            self._terminated = True
            # Flush stream
            # for packet in self.out_stream.encode():
            #     self.out_container.mux(packet)
            # Close the file
            #self.out_container.close()
            self.cv_out.release()
    def close(self):
        self.terminate()
        if self._was_temp:
            self.tempfile.close()
            #os.unlink(self.dst_obj)
            #self.dst_obj.close()
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def write_frame(self, np_frame):
        assert (np_frame.shape[0] == self.shape[0]) and (np_frame.shape[1] == self.shape[1]) and (np_frame.shape[2] == self.shape[2])
        # frame = av.VideoFrame.from_ndarray(np_frame, format="rgb24")
        # assert (frame.height == self.shape[0]) and (frame.width == self.shape[1])
        # print(f"The frame type and shape is {type(frame)}, {frame.shape}")
        
        #for packet in self.out_stream.encode(frame):
        #    self.out_container.mux(packet)
        bgr_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        self.cv_out.write(bgr_frame)
        self.finx+=1
    def ts_ms(self):
        return 1000 * (self.finx/self.fps)


# ********************************************************************************
    
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
