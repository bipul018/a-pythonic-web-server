import asyncio
import io
import av
import numpy
import torch
import cv2
import subprocess
import time
import tempfile
import pathlib
import fractions
from landmark import drawing_landmarks as landmark_drawer

import math


from video_rw.video_rw import *

# Given a list of frames to clip, which are in format [begin, end),
# Return a list of video bytes representing the clips
def clip_videos_frames(file_or_obj, clip_frames):
    with FrameGenStream(file_or_obj) as stream:
        (height, width, _) = stream.shape
        fps = stream.fps
        out_streams = []
        for _ in clip_frames:
            out_streams.append(VideoFromFrame(width=width, height=height, new_fps=fps))

        # For each frame generated, see if it lies in the range
        while (frame:=stream.next_frame()) is not None:
            for ((beg, end), ostr) in zip(clip_frames, out_streams):
                if (stream.finx >= beg) and (stream.finx < end):
                    ostr.write_frame(frame)

        outputs = []
        for ostr in out_streams:
            ostr.terminate()
            outputs.append(ostr.bytes())
            ostr.close()
        return outputs
    


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
    #print(f"The factor is {factor}")
    with FrameGenStream(file_or_obj) as in_stream:
        new_w = math.ceil(in_stream.shape[1]/factor)
        new_h = math.ceil(in_stream.shape[0]/factor)
        with VideoFromFrame(new_w, new_h, new_fps=in_stream.fps) as out_file:
            while (in_frame:=in_stream.next_frame()) is not None:
                #print(f"Shape of frame is {in_frame.shape}, of processed is {in_frame[::factor,::factor,:].shape}, when expected size is ({(new_h, new_w, 3)}")
                out_file.write_frame(in_frame[::factor,::factor,:])
            out_file.terminate()
            return out_file.bytes()

def draw_landmarks_on_video(file_or_obj):
    with FrameGenStream(file_or_obj) as in_stream:
        with VideoFromFrame(in_stream.shape[1], in_stream.shape[0], new_fps=in_stream.fps) as out_file:
            detector=landmark_drawer.load_detector()
            while (in_frame:=in_stream.next_frame()) is not None:
                try:
                    drawn_frame,_=landmark_drawer.run_on_image(detector, in_frame, int(in_stream.ts_ms()))
                except Exception as e:
                    print("Exception:", repr(e), " occured for drawing landmark on frame ", out_file.finx)
                    drawn_frame = in_frame
                out_file.write_frame(drawn_frame)
            out_file.terminate()
            return out_file.bytes()

import run_stsae_gcn        

def infer_stsae_prediction_on_video(file_or_obj):
    context_size = 20
    with FrameGenStream(file_or_obj, fix_frames=context_size) as in_stream:
        detector=landmark_drawer.load_detector()
        all_pts = torch.zeros((context_size,33,3)) #shape of input
        while (in_frame:=in_stream.next_frame()) is not None:
            # We dont care about exceptions here
            try:
                _,curr_pts=landmark_drawer.run_on_image(detector, in_frame, int(in_stream.ts_ms()))
                # TODO:: See if this .finx is correct
                all_pts[in_stream.finx] = torch.tensor(curr_pts)
            except Exception as e:
                print("Exception:", repr(e), " occured for drawing landmark on frame ", in_stream.finx)
                in_stream.finx -= 1
        # Infer the yoga and return a friendly string
        with torch.no_grad():
            inputs = all_pts.permute((2,0,1)).unsqueeze(0)
            outputs = run_stsae_gcn.model(inputs)
            maxval,pose_inx = torch.max(torch.softmax(outputs,1), 1)
            maxval = maxval.item() * 100
        return f"The predicted yoga pose is `{run_stsae_gcn.poses_list[pose_inx]} ({maxval:.1f}%)`" 
                
            
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

