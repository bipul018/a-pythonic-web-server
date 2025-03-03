import traceback
import io
import tempfile

import asyncio
import cv2
import numpy

from video_rw.video_rw import *

from video_tasks import clip_videos_frames

from .misc import *

if DEBUGGING_MODE:
    from matplotlib import pyplot as plot
    from debugging.things_comparator import visualize_3d_blazepose_comparison

# The video updating fxn from the main server file,
# This is to prevent the circular import as below
update_video_bytes = None
# from serv import update_video_bytes

from .segment import StreamingSegmentor
from .predict import try_make_predictor
from .landmark_stream import Landmark_Streamer

class ConnectionHandler:
    def __init__(self):
        self.videos = [] #A list of bytes representing video byte(s) ?
        self.curr_video_recorder = None # a VideoFromFrame() object
        self.replies = []
        # self.streaming_segmentor = None
        self.assumed_fps = 30
        self.streaming_segmentor = StreamingSegmentor()
        self.latest_active_predictor = None
        self.parallel_tasks = [] # Need to close these on finalizing connection

        self.landmark_streamer = Landmark_Streamer(reply_factor=2)
        pass

    def reset_streaming_segmentor(self):
        # Resets streaming segmentor only if it was not None in the first place
        if self.streaming_segmentor is not None:
            # dur = fps * 30(original frame count) / 30(original fps)
            self.streaming_segmentor = StreamingSegmentor(hold_duration=
                                                          self.assumed_fps*30/30)
    def new_data(self, metadata, bin_data):
        # For the first time, need to have a metadata of type 'A'??
        # Or should we make a dict and dispatch fxns directly

        # A message consisting of 'fps' argument is going to be a start of new everything
        if ('fps' in metadata) and ('width' in metadata) and ('height' in metadata):
            self.assumed_fps = metadata['fps']
            self.curr_video_recorder = VideoFromFrame(width=metadata['width'],
                                                      height=metadata['height'],
                                                      new_fps=metadata['fps'])
            self.reset_streaming_segmentor()
            print("New data added")
            pass

        # If the 'curr_video_recorder' is not null, and there is a frame_type member that is 'image/jpeg', then we record the frame [and also run frame by frame segmentation]
        if (self.curr_video_recorder is not None) and ('frame_type' in metadata) and (metadata['frame_type'] == 'image/jpeg'):
            np_image = cv2.imdecode(numpy.frombuffer(bin_data, numpy.uint8), cv2.IMREAD_COLOR)
            self.curr_video_recorder.write_frame(np_image)
            # Doing redundant work here, be careful
            landmark_reply = self.landmark_streamer.run_frame(np_image,
                                                              metadata['timestamp'])
            if landmark_reply:
                self.replies.append(landmark_reply)
            if self.streaming_segmentor is not None:
                # Also send some replies if mode has changed
                prev_mode = list(self.streaming_segmentor.get_history().items())[-1]
                self.streaming_segmentor.add_frame(np_image, metadata['timestamp'])
                new_mode = list(self.streaming_segmentor.get_history().items())[-1]

                # Here, reply with a serialized form of the keypoitns to draw,
                #   or reply with actual line coordinates to draw 

                if prev_mode[1][0] != new_mode[1][0]:
                    prev_state = prev_mode[1][0]
                    new_state = new_mode[1][0]
                    
                    predr = try_make_predictor(self.streaming_segmentor.get_history(), self.streaming_segmentor.features)
                    if predr is not None:
                        # TODO:: Need to make sure that this place was empty previously, and clean up resources accordingly if needed
                        self.latest_active_predictor = predr
                        pass
                    dprint(f"States changed at {metadata['timestamp']} from {prev_state} -> {new_state}")
                    self.replies.append({'timestamp': metadata['timestamp'],
                                         'message' : f'Yoga State Changed',
                                         'from' : f'{prev_state}',
                                         'to' : f'{new_state}',
                                         'frame_duration' : f'[{prev_mode[0]}, {prev_mode[0] + prev_mode[1][1]})',})
                    pass
                # Also update all other predictors from past (if they exist they wont be more than 3 at a time i think
                # TODO:: Since now the predictor can bre replaced before execution, if too short a hold phase, might fk things up
                if self.latest_active_predictor:
                    ans = self.latest_active_predictor.on_frame(self.streaming_segmentor.features)
                    if ans is not None:
                        ans['reply']['timestamp'] = metadata['timestamp']
                        ans['reply']['message'] = 'Yoga Predicted' 
                        self.replies.append(ans['reply'])

                        # Here launch the parallel_task predictor also
                        new_tasks = ans['tasks']
                        for t in new_tasks:
                            t.add_kwargs(timestamp=metadata['timestamp'])
                            self.parallel_tasks.append(t)
                            self.parallel_tasks[-1].launch()
                            pass # for each new_tasks

                        pass
                    if self.latest_active_predictor.isdone():
                        self.latest_active_predictor = None
                        pass
                    pass # if latest active predictor
                pass
            pass # Just received an element, now process it
        

        # A temporary testing measure, if sent 'clip_here', clip the video
        if (self.curr_video_recorder is not None) and ('clip_here' in metadata):
            (height, width, _) = self.curr_video_recorder.shape
            fps = self.curr_video_recorder.fps
            self.clip_video()
            self.curr_video_recorder = VideoFromFrame(width=width,
                                                      height=height,
                                                      new_fps=fps)
            self.reset_streaming_segmentor()
            pass

        # If the message contains 'get_clip' followed by an index, set that video
        if ('get_clip' in metadata) and (metadata['get_clip'] < len(self.videos)):
            update_video_bytes(self.videos[metadata['get_clip']])
            pass

        # If the message contains 'clip_count', return clip count
        if 'clip_count' in metadata:
            self.replies.append({'timestamp': metadata['timestamp'],
                                 'message': 'Clip Count Returned',
                                 'clip_count': len(self.videos)})
            pass

        # If the message contains 'last_frame' key (for now dont care about value), then finish recording and push new video [, then we run the video segmentation upon it]
        if ('last_frame' in metadata):
            self.clip_video()
            pass

        # If told to reset, just re-create the video recorder
        if ('reset' in metadata):
            (height, width, _) = self.curr_video_recorder.shape
            fps = self.curr_video_recorder.fps
            self.curr_video_recorder.close()
            self.curr_video_recorder = VideoFromFrame(width=width,
                                                      height=height,
                                                      new_fps=fps)
            self.reset_streaming_segmentor()
            pass
        
        self.handle_parallel_tasks()
        pass # self.new_data()

                                                     
    def force_close_tasks(self):
        for t in self.parallel_tasks:
            t.stop()
            pass

        self.parallel_tasks = None
        pass

    def handle_parallel_tasks(self):
        not_yet_complete = []
        for t in self.parallel_tasks:
            if not t.is_completed():
                not_yet_complete.append(t)
                pass
            else:
                # TODO:: Find if this step is compulsory or not
                t.wait_for_completion()
                # TODO:: This fxn might throw exception, for now letting it flow to top level
                res = t.get_result()
                if res['reply']:
                    self.replies.append(res['reply'])
                    pass
                for t_new in res['tasks']:
                    not_yet_complete.append(t_new)
                    pass
                pass
            pass # for t in self.parallel_tasks:
        self.parallel_tasks = not_yet_complete
        pass # self.handle_parallel_tasks()

    def pop_replies(self):
        # For now send any pose detection value back ??
        rep = self.replies
        self.replies = []
        return rep
        pass

    def clip_video(self):
        if self.curr_video_recorder is not None:
            self.curr_video_recorder.terminate()
            vid_bytes = self.curr_video_recorder.bytes()
            #self.videos.append(self.curr_video_recorder.bytes())
            self.curr_video_recorder.close()
            self.curr_video_recorder = None
            #update_video_bytes(self.videos[-1])
            # Split the video right here and return the data
            if self.streaming_segmentor is not None:
                debug_on_clip_reset(vid_bytes, self.streaming_segmentor)

            with tempfile.NamedTemporaryFile(mode='wb', suffix = ".mp4") as tfile:
                print(f"Writing file of length {len(vid_bytes)}")
                tfile.write(vid_bytes)
                if self.streaming_segmentor is not None:
                    states = self.streaming_segmentor.get_history()
                    # TODO:: fix the problem if it seems to affect operations
                    self.reset_streaming_segmentor()
                else:
                    states = temp_seg.segment_video(tfile.name) 
                print(f"The states are {states}")
                state_prints = []
                for s in states:
                    print(f"Next state is {s}: {states[s]}")
                    state_prints.append({s: f'{states[s]}'})
                    pass
                # TODO:: Need another way of handling what was going to be handled here
                #        Maybe a 'debug' message type??
                # self.replies.append({'clips' : state_prints})
                # Record the clips for the things also
                clip_ranges = [(i, i+states[i][1]) for i in states]
                clips = clip_videos_frames(io.BytesIO(vid_bytes), clip_ranges)
                _=[self.videos.append(c) for c in clips]
            print("Clipped the video")
        pass
        
    def on_close(self):
        self.force_close_tasks()
        #self.clip_video()
        if self.curr_video_recorder:
            self.curr_video_recorder.close()
            self.curr_video_recorder = None
            pass
        print("Closed connection")
        pass
 

# Only given a fk to when debugging
def debug_on_clip_reset(video_bytes, streaming_segmentor):
    if DEBUGGING_MODE:
        from matplotlib import pyplot as plot
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp4') as tfile:
            print(f"Writing file of length {len(vid_bytes)}")
            tfile.write(vid_bytes)
            states = streaming_segmentor.get_history()
            # Here the lagging video might actually cause the clipped video to lose 2 frames each clip
            (states_by_whole, feats_by_whole, vmag_by_whole) = temp_seg.segment_video(tfile.name, return_feats=True)
            print(f" State history:\n From streaming:{states}\n From whole video:{states_by_whole}")
            # Compare the differences in keypoints
            vmag_by_whole = vmag_by_whole.to(streaming_segmentor.velocity_mags.dtype)
                
            # pad the last two for streaming segmentor
            vmag_by_stream = streaming_segmentor.velocity_mags
            # use the deepseek generated fxn for comparing stuff
            vmag_comp = compare_tensors(vmag_by_whole, vmag_by_stream, epsilon=1e-3)

            print(f"Comparing velocity magnitudes : { {key: value for key, value in vmag_comp.items() if key not in ['all_diff_indices', 'padded1', 'padded2']} }")
                
            # Now plot the padded arrays
            plot.plot(vmag_comp['padded1'].numpy(), label='Whole at once')
            plot.plot(vmag_comp['padded2'].numpy(), label='Streaming mode')
            plot.legend()
            plot.show()
                
            # Now compare the original generated features also
            og_feats_stream = streaming_segmentor.features
            og_feats_whole = feats_by_whole.to(og_feats_stream.dtype)
            og_feats_comp = compare_tensors(og_feats_whole, og_feats_stream, epsilon=1e-3)
            print(f"Comparing original features : { {key: value for key, value in og_feats_comp.items() if key not in ['all_diff_indices', 'padded1', 'padded2']} }")
                
            visualize_3d_blazepose_comparison(og_feats_comp['padded1'].numpy(),
                                                  og_feats_comp['padded2'].numpy(),
                                                  interval=100)
            #tr_result = best_transform(source=og_feats_comp['padded2'].numpy().reshape(-1,3), target=og_feats_comp['padded1'].numpy().reshape(-1,3))
            #print(f"The best simple transformation that makes feats2(stream) -> feats1(whole) is {tr_result}")

    
