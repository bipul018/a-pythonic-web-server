from landmark import biomechanical_features as bio_feats
from landmark import temporal_segmentation as temp_seg
from landmark import keypoint_extractor as key_extr

import traceback
import io
import tempfile
import torch
import asyncio
import cv2
import numpy

from video_rw.video_rw import *

from tryvds import clip_videos_frames
import run_stsae_gcn


DEBUGGING_MODE = False

if DEBUGGING_MODE:
    # TODO:: Only a debugging thing, no need to commit this
    from debugging.things_comparator import compare_tensors
    from debugging.things_comparator import visualize_3d_blazepose_comparison
    from debugging.transform import best_transform
    from debugging.dynamic_plotter import DynamicPlotter
    from matplotlib import pyplot as plot

EXPECT_WEBCAM_ONLY = False

def dprint(*args, **kwargs):
    if DEBUGGING_MODE:
        print(*args, **kwargs)

def map_to_range(input_value, n, m):
    """Maps input_value from [0, n-1] to [0, m-1] using rounding."""
    return int(round((input_value / (n - 1)) * (m - 1)))  #Scale 


# Also will need a class that actually will do stuff given a list of features, and provide a function that can be triggered given a 'desired' destination frame number


# TODO:: Figure out how to receive video frames also later
#    Or just like taking in frame_keypoints, we will have to keep recording each frames' requirements
def try_make_predictor(state_history, curr_keypoints):
    # If eligible, make and return predictor, else None
    if (len(state_history) < 2) or (list(state_history.items())[-1][1][0] != temp_seg.PoseState.HOLD):
        return None
    dprint(f"===> Making a prediction object right now ...")
    return Predictor(state_history, curr_keypoints)
        

class Predictor:
    '''
    Will take in state history (not the segmentor), and the frame keypoints since the beginning.
    Will have some fxns and also the cases when it should be triggered ??
    Or have a fxn to be called on each frame, which decides stuff to do every time ??
    '''

    def __init__(self, state_history, curr_keypoints):
        assert len(state_history) >= 2
        # dont process until 3 more frames
        #  the end_point is the frame past the useful frame (i think)
        self.end_point = curr_keypoints.shape[0] + 3
        # Find the useful previous movement start
        # TODO:: make at least some assertions
        self.start_point = list(state_history.items())[-2][0]
        self.own_frames = None
        pass
    def isdone(self):
        return self.own_frames is not None
    def on_frame(self, curr_keypoints):
        if curr_keypoints.shape[0] >= self.end_point:
            self.own_frames = curr_keypoints[self.start_point:self.end_point]
            # now sample frames
            FRAME_COUNT = 20
            if self.own_frames.shape[0] < 20:
                # TODO:: Also indicate that it was recoverable error properly 
                return False
            sampled_frames = []
            for i in range(0, FRAME_COUNT):
                j = map_to_range(i, self.own_frames.shape[0], FRAME_COUNT)
                sampled_frames.append(self.own_frames[j])
                pass
            self.sampled_frames = torch.tensor(numpy.array(sampled_frames)).to(torch.float32)
            dprint(f"The type of tensor is {self.sampled_frames.dtype}")
            # reply with prediction
            inputs = self.sampled_frames.permute((2,0,1)).unsqueeze(0)
            outputs = run_stsae_gcn.model(inputs)
            # maxval,pose_inx = torch.max(torch.softmax(outputs,1), 1)
            # maxval = maxval.item() * 100
            maxvals, pose_inxs = torch.topk(torch.softmax(outputs,1), k=4, dim=1)
            maxvals = [round(v.item(), 2) for v in list(maxvals.squeeze() * 100)]
            names = [[run_stsae_gcn.poses_list[idx] for idx in row] for row in pose_inxs]

            # calculate the suggestions
            suggestion = "Just enjoy your life"
            dprint(f"Predicted {suggestion} @ {names[0]}")
            # Outside of this fxn, if suggestion received, send reply 

            # For now reply also a audio 

            # return the suggeestions
            # Observation: Only one of these can be active at a time anyway
            # so maybe just keep it as a service akways on
            # Would help more as it would be a time consuming process to do in reality
            return { 'Yoga Poses'  : names,
                     'Confidences' : maxvals,
                     'Suggestion' : suggestion }
        return None
                     
            

# A class that helps one partition the video into states
class StreamingSegmentor:
    def __init__(self, movement_threshold=0.3, hold_threshold=0.1, hold_duration=30):
        self.pose_detector = key_extr.PoseExtractor(rgb_mode=True)
        self.feature_count = 33
        self.machine_args = {'movement_threshold': movement_threshold,
                             'hold_threshold': hold_threshold,
                             'hold_duration': hold_duration}
        self.state_machine = temp_seg.YogaPoseStateMachine(
            movement_threshold=self.machine_args['movement_threshold'],
            hold_threshold=self.machine_args['hold_threshold'],
            hold_duration=self.machine_args['hold_duration'] 
        )
        # The last two features are to be replaced always ??
        self.extractor = bio_feats.BiomechanicalFeatureExtractor()
        self.features = torch.empty(0, self.feature_count, 3)
        # The velocity_mags will lag behind features by 2 frames,
        #    so at end of segmentation, just have to update by extra 2 zero values for state machine
        # TODO:: Need to decide what this means
        self.velocity_mags = torch.tensor([])
        pass



    def add_frame(self, np_frame, ts_ms):
        # Extract features
        # If you cannot detect features, for now push the latest feature ?? or a zeroed out feature ?
        # TODO:: Asses what is wrong and what is right based on what aagab does
        try:
            _, new_feats = self.pose_detector.process_frame(np_frame, also_annotate=False)
            if new_feats is None:
                raise Exception("No landmarks")
            # Doing this here because no landmarks were found for this frame, might have to make this configurable later on
            return self.add_feature(new_feats)
        except Exception as e:
            print(f"**** There was no landmark detected for ts {ts_ms} because `{e}`. Skipping from feature array ****")
            print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
            # new_feats = numpy.zeros((self.feature_count, 3))
            pass
        pass


    def add_feature(self, new_feats):

        self.features = torch.cat((self.features, torch.tensor(new_feats).reshape(1, self.feature_count, 3)))


        # if 3 frames have not been collected yet, skip
        if self.features.shape[0] >= 3:
            # Take the last three features and do the velocity magnitude calculation
            velocity = self.extractor.extract_features(self.features[-3:,:])["Joint Acceleration"]
    
            v = torch.sqrt(velocity[..., 0]**2 + velocity[...,1]**2 + velocity[...,2]**2)
            velocity_magnitude = v.sum(dim=-1) 
            # velocity_magnitude = velocity_magnitude
            velocity_magnitude = velocity_magnitude.clamp_(min=0, max=1.5).pow_(2).clamp_(max=1.5)

            # print(f"The shapes involved are : features->{self.features.shape}, new_feats->{new_feats.shape}, velocity_magnitude->{velocity_magnitude.shape}, mags->{self.velocity_mags.shape}")
            # Take the first value only and push it
            self.velocity_mags = torch.cat((self.velocity_mags, torch.tensor([velocity_magnitude[0].item()], dtype=self.features.dtype)))
            self.state_machine.process_frame(self.velocity_mags[-1])
        pass
    def get_history(self):
        # TODO::This might need to forward the frames by 2 , because the state machine was lagging behind by two
        return self.state_machine.get_state_history()


