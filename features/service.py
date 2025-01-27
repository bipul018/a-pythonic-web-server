from . import biomechanical_features as bio_feats
from . import temporal_segmentation as temp_seg
from . import keypoint_extractor as key_extr

import torch
import draw_landmarks

EXPECT_WEBCAM_ONLY = False

        
        

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

    
