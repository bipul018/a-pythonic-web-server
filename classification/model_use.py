import torch
import os
from .new_STSAE_GCN import STSAE_GCN

from .yoga_pose_target_data import poses as poses_list

# Loading the model code 
#model_path = os.path.join("./some_models", "dec16_cross_fold_best_model.pth")
#model_path = os.path.join('./classification', 'STSAE_GCN.pth')
model_path = os.path.join('./classification', 'final_model_again.pth')

# TODO:: If possible figure out this from some other places directly
in_channels = 3
#hidden_channels= 64
hidden_channels= 120
num_classes= len(poses_list)
num_frames= 20
#num_blocks = 9
num_blocks = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STSAE_GCN(in_channels, hidden_channels, num_classes, num_frames, num_blocks) 
model_saved_state = torch.load(model_path, weights_only = True,
                        map_location = device)
#model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(model_saved_state)

if model is None:
    raise Exception(f"Did not find any model to load from checkpoint")
else:
    print(f"Found the model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
