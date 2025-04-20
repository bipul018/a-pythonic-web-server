joint_configs = {
        # Upper body
        'right_shoulder': {
            'joints': (13, 11, 23),  # right_elbow, right_shoulder, right_hip
            'joint_names': ('right_elbow', 'right_shoulder', 'right_hip'),
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_shoulder': {
            'joints': (14, 12, 24),  # left_elbow, left_shoulder, left_hip
            'joint_names': ('left_elbow', 'left_shoulder', 'left_hip'),
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'right_elbow': {
            'joints': (11, 13, 15),  # right_shoulder, right_elbow, right_wrist
            'joint_names': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_elbow': {
            'joints': (12, 14, 16),  # left_shoulder, left_elbow, left_wrist
            'joint_names': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        
        # Lower body
        'right_hip': {
            'joints': (11, 23, 25),  # right_shoulder, right_hip, right_knee
            'joint_names': ('right_shoulder', 'right_hip', 'right_knee'),
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_hip': {
            'joints': (12, 24, 26),  # left_shoulder, left_hip, left_knee
            'joint_names': ('left_shoulder', 'left_hip', 'left_knee'),
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'right_knee': {
            'joints': (23, 25, 27),  # right_hip, right_knee, right_ankle
            'joint_names': ('right_hip', 'right_knee', 'right_ankle'),
            'planes': ['sagittal', 'transverse', 'frontal']  # Now including transverse
        },
        'left_knee': {
            'joints': (24, 26, 28),  # left_hip, left_knee, left_ankle
            'joint_names': ('left_hip', 'left_knee', 'left_ankle'),
            'planes': ['sagittal', 'transverse', 'frontal']  # Now including transverse
        },
        'right_ankle': {
            'joints': (25, 27, 31),  # right_knee, right_ankle, right_foot_index
            'joint_names': ('right_knee', 'right_ankle', 'right_foot_index'),
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_ankle': {
            'joints': (26, 28, 32),  # left_knee, left_ankle, left_foot_index
            'joint_names': ('left_knee', 'left_ankle', 'left_foot_index'),
            'planes': ['sagittal', 'transverse', 'frontal']
        }
}

poses = ['downward-dog',
         'standing-forward-bend',
         'half-way-lift',
         'mountain',
         'chair',
         'cobra',
         'cockerel',
         'extended-triangle',
         'extended-side-angle',
         'corpse',
         'staff',
         'wind-relieving',
         'fish']

# Breathing guidance for the hold state of each pose
POSE_BREATHING_HOLD_STATE = {
    'downward-dog': "Deep steady breaths",
    'standing-forward-bend': "Relaxed deep breaths",
    'half-way-lift': "Steady breaths", # Often transitional, but steady if held
    'mountain': "Calm even breaths",
    'chair': "Strong steady breaths",
    'cobra': "Steady breaths",
    'cockerel': "Focused breaths", # Assuming a balancing/squat pose
    'extended-triangle': "Expansive breaths",
    'extended-side-angle': "Deep steady breaths",
    'corpse': "Natural relaxed breaths",
    'staff': "Calm steady breaths",
    'wind-relieving': "Relaxed breaths",
    'fish': "Deep chest breaths"
}
pose_list = ['downward-dog','standing-forward-bend','half-way-lift',
             'mountain','chair','cobra','cockerel','extended-triangle',
             'extended-side-angle','corpse','staff','wind-relieving','fish'
            ]
ACTION_JOINT_MAPPING = {
    'downward-dog': ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_elbow', 'left_elbow'],
    'standing-forward-bend': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow'],
    'half-way-lift': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'],
    'mountain': ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_knee', 'left_knee'],
    'chair': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'],
    'cobra': ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_elbow', 'left_elbow'],
    'cockerel': ['right_elbow', 'left_elbow', 'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle'],
    'extended-triangle': ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_knee', 'left_knee'],
    'extended-side-angle': ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_knee', 'left_knee'],
    'corpse': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'],
    'staff': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'],
    'wind-relieving': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'],
    'fish': ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_elbow', 'left_elbow']
}

# Dictionary mapping poses to joints, ordered by perceived importance for feedback
PRIORITY_JOINT_MAPPING = {
    'downward-dog': ['right_hip', 'left_hip', 'right_shoulder', 'left_shoulder', 'right_knee', 'left_knee', 'right_elbow', 'left_elbow'], # Hips/Shoulders for V shape, then leg/arm extension
    'standing-forward-bend': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow'], # Hip flexion primary, then knees, ankles, relaxed arms last
    'half-way-lift': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'], # Hip angle for flat back, then straight knees, then shoulder alignment
    'mountain': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'], # Neutral hips/knees for straightness, then relaxed shoulders
    'chair': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'], # Hip/Knee bend primary, then shoulder flexion for arms up
    # MAYBE ELBOW MORE IMPORTANT FOR COBRA
    'cobra': ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_elbow', 'left_elbow'], # Shoulder/Hip extension for backbend, then elbow support
    'cockerel': [ 'right_elbow', 'left_elbow', 'right_hip', 'left_hip','right_knee', 'left_knee', 'right_ankle', 'left_ankle'], # Hips/Knees/Ankles for balance/squat, then arms
    'extended-triangle': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'], # Hip opening/rotation, straight legs, then arm extension/alignment
    'extended-side-angle': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'], # Hip/Knee angles for lunge, then shoulder/arm alignment
    'corpse': ['right_hip', 'left_hip', 'right_shoulder', 'left_shoulder', 'right_knee', 'left_knee'], # Overall relaxation, focus on major joints lying flat/neutral
    'staff': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'], # Hip angle for seated upright, straight knees, then shoulder posture
    'wind-relieving': ['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder'], # Hip/Knee flexion primary, relaxed shoulders
    'fish': ['right_elbow', 'left_elbow','right_shoulder', 'left_shoulder',  'right_hip', 'left_hip'] # Shoulder/Elbow support for backbend, then hip extension
}



# THE VALUES AREN'T ACCURATE; AKA PLACEHOLDER.
TARGET_ANGLES = {
    'downward-dog': {
        'right_shoulder': 180,  # Arms should be straight and aligned with the back.
        'left_shoulder': 180,
        'right_hip': 90,       # Hips should form a right angle with legs.
        'left_hip': 90,
        'right_knee': 180,     # Legs should be straight.
        'left_knee': 180,
        'right_elbow': 180,    # Elbows should be straight
        'left_elbow': 180      # Elbows should be straight
    },
    'standing-forward-bend': {
        'right_hip': 45,      # Hip flexion with slight anterior pelvic tilt
        'left_hip': 45,
        'right_knee': 175,     # Microbend to protect hamstrings
        'left_knee': 175,
        'right_ankle': 85,     # Dorsiflexion for weight distribution
        'left_ankle': 85,
        'right_shoulder': 160, # Shoulder protraction with arm hang
        'left_shoulder': 160,
        'right_elbow': 170,    # Gentle bend to avoid hyperextension
        'left_elbow': 170
    },
    'half-way-lift': {
        'right_hip': 90,       # Flat back requires hips at 90 degrees.
        'left_hip': 90,
        'right_knee': 180,     # Knees are straight.
        'left_knee': 180,
        'right_shoulder': 90,  # Shoulders align with the back, creating a 90-degree angle with the arms.
        'left_shoulder': 90
    },
    'mountain': {
        'right_shoulder': 0,   # Arms hang naturally down the sides.
        'left_shoulder': 0,
        'right_hip': 180,      # Hips in a neutral position, no flexion or extension.
        'left_hip': 180,
        'right_knee': 180,     # Legs are straight.
        'left_knee': 180
    },
    'chair': {
        'right_hip': 120,      # Hips are bent, but not a full 90 degrees.
        'left_hip': 120,
        'right_knee': 120,     # Knees are bent, aligning with hips.
        'left_knee': 120,
        'right_shoulder': 180, # Arms are fully extended upward.
        'left_shoulder': 180
    },
    'cobra': {
        'right_shoulder': 30, # Arms are straight, pushing the upper body up.
        'left_shoulder': 30,
        'right_hip': 180,      # Hips are pressed into the ground, no flexion.
        'left_hip': 180,
        'right_elbow': 180,      # Elbows are straight.
        'left_elbow': 180
    },
    'cockerel': {
        'right_elbow': 180,     # Elbows are bent to hold the body.
        'left_elbow': 180,
        'right_hip': 45,       # Hips are flexed to lift the legs.
        'left_hip': 45,
        'right_knee': 30,      # Knees are bent to compact the pose.
        'left_knee': 30,
        'right_ankle': 180,     # Ankles are flexed for balance.
        'left_ankle': 180
    },
    'extended-triangle': {
        'right_shoulder': 90, # Top arm is extended upward.
        'left_shoulder': 90,  # Bottom arm points downward.
        'right_hip': 90,       # Hips are bent sideways.
        'left_hip': 90,
        'right_knee': 180,     # Front leg is straight.
        'left_knee': 180
    },
    'extended-side-angle': {
        'right_shoulder': 130,  # Top arm extends over the ear.
        'left_shoulder': 90,   # Bottom arm rests on the bent leg or floor.
        'right_hip': 180,       # Hips are deeply bent.
        'left_hip': 30,
        'right_knee': 180,      # Front leg is bent at 90 degrees.
        'left_knee': 90       # Back leg is straight.
    },
    'corpse': {
        'right_hip': 180,      # Full relaxation, neutral position.
        'left_hip': 180,
        'right_knee': 180,     # Legs are straight.
        'left_knee': 180,
        'right_shoulder': 0,   # Arms rest at the sides.
        'left_shoulder': 0
    },
    'staff': {
        'right_hip': 90,       # Hips are at a right angle for the seated position.
        'left_hip': 90,
        'right_knee': 180,     # Legs are straight.
        'left_knee': 180,
        'right_shoulder': 30,  # Shoulders align with the back.
        'left_shoulder': 30
    },
    'wind-relieving': {
        'right_hip': 45,       # Hips are flexed to bring the knees toward the chest.
        'left_hip': 45,
        'right_knee': 35,     # Knees are bent.
        'left_knee': 35,
        'right_shoulder': 30,   # Arms wrap around the knees.
        'left_shoulder': 30
    },
    'fish': {
        'right_shoulder': 90,  # Arms support the body at a 90-degree angle.
        'left_shoulder': 90,
        'right_hip': 180,      # Hips remain neutral.
        'left_hip': 180,
        'right_elbow': 90,     # Elbows are bent to support the chest lift.
        'left_elbow': 90
    }
}
