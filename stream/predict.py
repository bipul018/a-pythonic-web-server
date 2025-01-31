from landmark import biomechanical_features as bio_feats
from landmark import temporal_segmentation as temp_seg
from landmark import keypoint_extractor as key_extr
from tts.text_to_speech import text_to_speech

import tempfile
import base64
import io
import torch
import asyncio
import numpy

from classification import model_use as stsae_gcn
from coaching.feedback import generate_pose_feedback

from .misc import map_to_range, dprint, DEBUGGING_MODE

if DEBUGGING_MODE:
    from debugging.timeouters import setup_timeout, reset_timeout
    pass


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
                return {}
            sampled_frames = []
            for i in range(0, FRAME_COUNT):
                j = map_to_range(i, self.own_frames.shape[0], FRAME_COUNT)
                sampled_frames.append(self.own_frames[j])
                pass
            self.sampled_frames = torch.tensor(numpy.array(sampled_frames)).to(torch.float32)
            dprint(f"The type of tensor is {self.sampled_frames.dtype}")
            # reply with prediction
            inputs = self.sampled_frames.permute((2,0,1)).unsqueeze(0)
            outputs = stsae_gcn.model(inputs)
            # maxval,pose_inx = torch.max(torch.softmax(outputs,1), 1)
            # maxval = maxval.item() * 100
            maxvals, pose_inxs = torch.topk(torch.softmax(outputs,1), k=4, dim=1)
            maxvals = [round(v.item(), 2) for v in list(maxvals.squeeze() * 100)]
            names = [[stsae_gcn.poses_list[idx] for idx in row] for row in pose_inxs]

            # calculate the suggestions
            suggestion = f"You are doing {names[0][0]} pose. Never ever try to make python your first choice. Python is shit!!!!"
            suggestion = f"You are doing {names[0][0]} pose. Hello this beautiful world where people can do yoga with ai assistance!!!!!"

            angles_dict, _ = bio_feats.calculate_joint_angles(self.sampled_frames)
            suggestion = generate_pose_feedback(angles_dict, names[0][0])

            dprint(f"Predicted {suggestion} @ {names[0]}")
            # Outside of this fxn, if suggestion received, send reply 
            # bio_feats.calculate_joint_angles(self.own_frames)

            #if DEBUGGING_MODE:
            #    setup_timeout(10)
            #    pass
                
            # For now reply also a audio 
            # with tempfile.NamedTemporaryFile(mode='rb', suffix='.wav') as tfile:
            #     text_to_speech(suggestion, file_or_name = tfile.name)
            #     vbytes = tfile.read()
            #     print(f"The output message bytes is of length {len(vbytes)}")
            #     output_sound = base64.b64encode(vbytes).decode('utf-8')
            #     pass
            # print(f"The output message in form of base64 voice if of length {len(output_sound)}")
            
            #if DEBUGGING_MODE:
            #    reset_timeout(10)
            #    pass

            # return the suggeestions
            # Observation: Only one of these can be active at a time anyway
            # so maybe just keep it as a service akways on
            # Would help more as it would be a time consuming process to do in reality
            return { 'poses'  : names,
                     'confidences' : maxvals,
                     'text_suggestion' : suggestion, }
                     #'voice_suggestion' : output_sound }
        return None
                     
            
