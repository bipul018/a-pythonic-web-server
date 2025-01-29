# ---------------------------------------------------------------------------------------------
#    Need to write the minimal overhead code above to make this code below work
# ---------------------------------------------------------------------------------------------
poses_list = [
    'Downward Dog',
    'Standing Forward Bend',
    'Half Way Lift',
    'Mountain',
    'Chair',
    'Cobra',
    'Cockerel',
    'Extended Triangle',
    'Extended Side Angle',
    'Corpse',
    'Staff',
    'Wind Relieving',
    'Fish'
]

# Loading the model code 
#checkpoint_path = os.path.join(stsae_gcn.SAVE_PATH, 'best_model.pth')
checkpoint_path = os.path.join("./some_models", "dec16_cross_fold_best_model.pth")
in_channels = 3
hidden_channels= 64
#num_classes= stsae_gcn.NUM_CLASSES
num_classes= len(poses_list)
num_frames= 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STSAE_GCN(in_channels, hidden_channels, num_classes, num_frames) 
checkpoint = torch.load(checkpoint_path, weights_only = False,
                        map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])

if model is None:
    raise Exception(f"Did not find any model to load from checkpoint")
else:
    print(f"Found the model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ---------------------------------------------------------------------------------------------
#      Functions to give the model prediction given the whole video frames 
# ---------------------------------------------------------------------------------------------


        

        



