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

    
