#from .STSAE_GCN import STSAE_GCN
from .new_STSAE_GCN import STSAE_GCN
from .yoga_pose_target_data import poses as poses_list
import os
import torch
import torch.nn as nn
# Define constants
NUM_CHANNELS = 3
NUM_CLASSES = len(poses_list)
NUM_FRAMES = 20
NUM_BLOCKS = 5
HIDDEN_CHANNELS = 120
model_path = os.path.join('./classification', 'best_model_fold_None.pth')


# Load the pre-trained model weights
# model.load_state_dict(torch.load(model_path,map_location='cpu', weights_only=True))
def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and training state from a checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only = False,map_location='cpu')

    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Get the epoch number to resume from
    start_epoch = checkpoint['epoch']

    # Load training history with new metrics
    history = checkpoint.get('history', {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    })

    return model, optimizer, start_epoch, history



# Set the number of threads to 1
torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STSAE_GCN(NUM_CHANNELS, HIDDEN_CHANNELS, NUM_CLASSES, NUM_FRAMES, num_blocks=NUM_BLOCKS)
# model = STSAE_GCN(3, HIDDEN_CHANNELS, NUM_CLASSES, 20,num_blocks=NUM_BLOCKS)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if model_path and os.path.exists(model_path):
    model, optimizer, start_epoch, history = load_checkpoint(
        model, optimizer, model_path
    )
    print(f'Model is ------\n{model}')


if model is None:
    raise Exception(f"Did not find any model to load from checkpoint")
else:
    print(f"Found the model")
    print(model)
    print(model.fc.weight)
    print('fdsfasdTesting on a random input')
    print(model(torch.rand((2,3,20,33))).shape)
    print('fdsfasdTested on a random input')    
model = model.to(device)
model.eval()
