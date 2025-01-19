from . import biomechanical_features as bio_feats
from . import temporal_segmentation as temp_seg
from . import keypoint_extractor as key_extr

import torch
import draw_landmarks

EXPECT_WEBCAM_ONLY = False

# Context needed for the making the object
class PerFrameDetector:
    def __init__(self, movement_threshold=0.3, hold_threshold=0.1, hold_duration=30):
        self.p_detector = draw_landmarks.load_detector()
        self.state_machine = temp_seg.YogaPoseStateMachine(
            movement_threshold=movement_threshold,
            hold_threshold=hold_threshold,
            hold_duration=hold_duration 
        )
        self.extractor = bio_feats.BiomechanicalFeatureExtractor()
        
        self.features = torch.empty(0, 33, 3)
        # self.velocities = []
    def add_frame(self, rgb_image, ts_ms):
        # TODO:: Might be the case that this requires float type type, but we have int type
        
        _, new_feats = draw_landmarks.run_on_image(self.p_detector, rgb_image, ts = ts_ms, also_draw=False)
        self.features = torch.cat((self.features, torch.tensor(new_feats).reshape(1, 33,3)))
        # Only do velocity extraction if >= 3 frames collected
        if self.features.shape[0] >= 3:
            velocity = self.extractor.extract_features(self.features)["Joint Acceleration"]
            # TODO:: Need to prevent doing this redundant calculation
            v = torch.sqrt(velocity[..., 0]**2 + velocity[...,1]**2 + velocity[...,2]**2)
            velocity_magnitude = v.sum(dim=-1) 
            velocity_magnitude = velocity_magnitude.clamp_(min=0, max=1.5).pow_(2).clamp_(max=1.5)
            # But if len is == 3, need to also give all to state machine
            if len(self.features) == 3:
                for fv in velocity_magnitude[:-1]:
                    self.state_machine.process_frame(fv)
            self.state_machine.process_frame(velocity_magnitude[-1])
        return self.state_machine.state
