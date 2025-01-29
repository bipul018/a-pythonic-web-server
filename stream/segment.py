from landmark import biomechanical_features as bio_feats
from landmark import temporal_segmentation as temp_seg
from landmark import keypoint_extractor as key_extr

import traceback
import torch
import cv2
import numpy

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


