import asyncio
import io
import av
import numpy
import cv2
import subprocess
import time
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

class VideoFromFrame:
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
        for packet in self.out_stream.encode(frame):
            self.out_container.mux(packet)
        self.finx+=1
    def ts_ms(self):
        return 1000 * (self.finx/self.fps)

class FrameGenStream:
    def __init__(self, file_or_obj,
                 fix_fps = None, fix_frames = None):
        self.in_container = av.open(file_or_obj, mode='r')
        vid_stream = self.in_container.streams.video[0]
        vid_stream.thread_type = "AUTO" # makes it go faster
        self.shape = (int(vid_stream.height), int(vid_stream.width), 3)
        self.max_frames = int(vid_stream.frames)
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
    
def draw_video(file_or_obj, fixed_frames = None, fixed_fps = None):
    got_frames = 0
    with FrameGenStream(file_or_obj, fix_fps=fixed_fps, fix_frames=fixed_frames) as stream:
        while (frame:=stream.next_frame()) is not None:
            cv2.imshow('Video Stream', numpy.flip(frame, axis=-1))
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(stream.duration/stream.desired_frames)
            got_frames += 1
        cv2.destroyAllWindows()
    print(f"Frames got = {got_frames}, width = {stream.shape[1]}, height = {stream.shape[0]}, frames = {stream.max_frames}, Desired frames = {stream.desired_frames}, duration = {stream.duration}")

def downsample_it(file_or_obj, factor=2):
    with FrameGenStream(file_or_obj) as in_stream:
        with VideoFromFrame(math.ceil(in_stream.shape[1]/factor), math.ceil(in_stream.shape[0]/factor), new_fps=in_stream.fps) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                out_file.write_frame(in_frame[::factor,::factor,:])
            out_file.terminate()
            return out_file.bytes()

def draw_landmarks_on_video(file_or_obj):
    with FrameGenStream(file_or_obj) as in_stream:
        with VideoFromFrame(in_stream.shape[1], in_stream.shape[0], new_fps=in_stream.fps) as out_file:
            detector=draw_landmarks.load_detector()
            while (in_frame:=in_stream.next_frame()) is not None:
                try:
                    drawn_frame,_=draw_landmarks.run_on_image(detector, in_frame, int(in_stream.ts_ms()))
                except Exception as e:
                    print("Exception:", repr(e), " occured for drawing landmark on frame ", out_file.finx)
                    drawn_frame = in_frame
                out_file.write_frame(drawn_frame)
            out_file.terminate()
            return out_file.bytes()
            
def sample_at_fps(file_or_obj, fps):
    with FrameGenStream(file_or_obj, fix_fps = fps) as in_stream:
        with VideoFromFrame(in_stream.shape[1], in_stream.shape[0], new_fps=fps) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                out_file.write_frame(in_frame)
            out_file.terminate()
            return out_file.bytes()

def query_info(file_or_obj):
    with FrameGenStream(file_or_obj) as vid:
        return {"width" : vid.shape[1],
                "height" : vid.shape[0],
                "duration" : vid.duration,
                "fps" : vid.fps,
                "frames" : vid.max_frames}
        

def sample_n_frames(file_or_obj, frames):
    with FrameGenStream(file_or_obj, fix_frames = frames) as in_stream:
        print(f"New video framerate after selecting{frames} frames is {frames/in_stream.duration}")
        with VideoFromFrame(in_stream.shape[1], in_stream.shape[0], new_fps=frames/in_stream.duration) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                out_file.write_frame(in_frame)
            out_file.terminate()
            return out_file.bytes()

