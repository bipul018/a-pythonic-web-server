import asyncio
import ffmpeg
import av
import numpy
import cv2
import subprocess
import time
import tempfile
import pathlib
import fractions
# ffmpeg.probe() contains 'streams' and 'format'
# a stream contains 'codec_type' which is either video or audio
#    also contains width and height, also contains 'nb_frames' which gives total number of franes
#    and 'duration' is the length of video in seconds

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
    def __init__(self, width, height, new_fps, out_name = None):
        if out_name is None:
            tf = tempfile.NamedTemporaryFile(suffix=".mp4",delete=False)
            self.file_name = tf.name
            tf.close()
            self._was_temp = True
        else:
            self._was_temp = False
            self.file_name = out_name
        self.shape = (height, width, 3)
        self.out_container = av.open(self.file_name, mode='w')
        frac_fps = fractions.Fraction(new_fps).limit_denominator(10000)
        self.out_stream = self.out_container.add_stream('mpeg4', rate=frac_fps)
        self.out_stream.width = width
        self.out_stream.height = height
        self.finx = 0
        self.fps = new_fps
        self._terminated = False
    def __enter__(self):
        return self
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
            pathlib.Path(self.file_name).unlink()
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def write_frame(self, np_frame):
        assert (np_frame.shape[0] == self.shape[0]) and (np_frame.shape[1] == self.shape[1]) and (np_frame.shape[2] == self.shape[2])
        #self.process.stdin.write(np_frame.astype(numpy.uint8).tobytes())
        frame = av.VideoFrame.from_ndarray(np_frame, format="rgb24")
        assert (frame.height == self.shape[0]) and (frame.width == self.shape[1])
        for packet in self.out_stream.encode(frame):
            self.out_container.mux(packet)
        self.finx+=1
    def ts_ms(self):
        return 1000 * (self.finx/self.fps)

class FrameGenStream:
    def __init__(self, file_name,
                 fix_fps = None, fix_frames = None):
        #self.process, self.width, self.height, self.max_frames, self.duration = make_video_stream(file_name)
        args = (
            ffmpeg
            .input(file_name)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .compile()
        )
        probe = ffmpeg.probe(file_name)
        stream = find_first_match(probe['streams'], 'codec_type', 'video')
        self.shape = (int(stream['height']), int(stream['width']), 3)
        self.max_frames = int(stream['nb_frames'])
        self.duration = float(stream['duration'])
        self.fps = self.max_frames/self.duration
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE)
        self.total_size = self.shape[0] * self.shape[1] * self.shape[2]
        # calculate the desired fps to capture at/interpolate at
        if fix_frames is not None:
            self.desired_frames = fix_frames
        elif fix_fps is not None:
            self.desired_frames = math.floor(fix_fps * self.duration)
        else:
            self.desired_frames = self.max_frames
        if self.desired_frames > self.max_frames:
            raise f"Desired frames is {self.desired_frames} which is greater than total frames that is {self.max_frames}"
        self.finx = -1
        self.actual_frame = 0
    def __enter__(self):
        return self
    def close(self):
        self.process.terminate()
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def next_frame(self):
        if self.actual_frame >= self.max_frames:
            return None
        old_finx = self.finx
        in_bytes = None
        while old_finx == self.finx:
            in_bytes = None
            in_bytes = self.process.stdout.read(self.total_size)
            if not in_bytes:
                in_bytes = None
                break
            self.actual_frame += 1
            self.finx = map_to_range(self.actual_frame,
                                     self.max_frames,
                                     self.desired_frames)
        if in_bytes is not None:
            return numpy.frombuffer(in_bytes, numpy.uint8).reshape(self.shape)
        return None
    def ts_ms(self):
        return 1000 * (self.actual_frame / self.fps)
    
def draw_video(file_name, fixed_frames = None, fixed_fps = None):
    #process, width, height, frames, duration = make_video_stream(file_name)
    got_frames = 0
    #for frame in get_video_frame(process.stdout, width, height, frames):
    print(f'File {file_name} is going to be played')
    with FrameGenStream(file_name, fix_fps=fixed_fps, fix_frames=fixed_frames) as stream:
        while (frame:=stream.next_frame()) is not None:
            cv2.imshow('Video Stream', numpy.flip(frame, axis=-1))
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(stream.duration/stream.desired_frames)
            got_frames += 1
        cv2.destroyAllWindows()
    print(f"Frames got = {got_frames}, width = {stream.shape[1]}, height = {stream.shape[0]}, frames = {stream.max_frames}, Desired frames = {stream.desired_frames}, duration = {stream.duration}")
    
infile = '/home/weepingcoder/fks/vdos/one-downdog.mp4'

def test_downsampling(filename):
    with FrameGenStream(filename) as in_stream:
        with VideoFromFrame(math.ceil(in_stream.shape[1]/2), math.ceil(in_stream.shape[0]/2), new_fps=in_stream.fps) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                out_file.write_frame(in_frame[::2,::2,:])
            out_file.terminate()
            with FrameGenStream(out_file.file_name) as processed_stream:
                while (p_frame:=processed_stream.next_frame()) is not None:
                    cv2.imshow('Downsampled Stream', numpy.flip(p_frame, axis=-1))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    time.sleep(processed_stream.duration/processed_stream.desired_frames)
                cv2.destroyAllWindows()

def downsample_it(filename, factor=2):
    with FrameGenStream(filename) as in_stream:
        with VideoFromFrame(math.ceil(in_stream.shape[1]/factor), math.ceil(in_stream.shape[0]/factor), new_fps=in_stream.fps) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                out_file.write_frame(in_frame[::factor,::factor,:])
            out_file.terminate()
            print(f"Terminated downsampling {filename} in {out_file.file_name}")
            with open(out_file.file_name, 'rb') as f:
                return f.read()

def sample_at_fps(filename, fps):
    with FrameGenStream(filename, fix_fps = fps) as in_stream:
        with VideoFromFrame(in_stream.shape[1], in_stream.shape[0], new_fps=fps) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                out_file.write_frame(in_frame)
            out_file.terminate()
            print(f"Terminated setting to {fps} fps of {filename} to {out_file.file_name}")
            with open(out_file.file_name, 'rb') as f:
                return f.read()
def query_info(filename):
    with FrameGenStream(filename) as vid:
        return {"width" : vid.shape[1],
                "height" : vid.shape[0],
                "duration" : vid.duration,
                "fps" : vid.fps,
                "frames" : vid.max_frames}
        

def sample_n_frames(filename, frames):
    with FrameGenStream(filename, fix_frames = frames) as in_stream:
        print(f"New video framerate after selecting{frames} frames is {frames/in_stream.duration}")
        with VideoFromFrame(in_stream.shape[1], in_stream.shape[0], new_fps=frames/in_stream.duration) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                out_file.write_frame(in_frame)
            out_file.terminate()
            print(f"Terminated selecting {frames} frames of {filename} to {out_file.file_name}")
            with open(out_file.file_name, 'rb') as f:
                return f.read()


# Now try to make a file from a bytestream
def test_from_bytes(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        with tempfile.NamedTemporaryFile(mode='wb') as infile:
            print(f"Actual file is {filename}, file size is {len(data)}, temp file is {infile.name}")
            infile.write(data)
            draw_video(infile.name)


            

# Set up a video processing service

# Takes in a json with two keys, service-id and args
# service-id is the name of the service and args is a list of args to pass to that service
# Returns either error or success, as a json with two keys, return-status return-value
# if return-status is error, return-value will have the relevant description or dump
# else return-status will be success and return-value will have whatever was returned by fxn
# The arguments is a list of simple {type, data} pairs of serialized items (for now in json)
# if the type is 'video' then the data will be, for now base64 encoded, blob of video file





