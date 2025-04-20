import torch
from typing import Dict, Tuple, Any, Union
from .Language_Model import generate_short_prompt as gen_prompt
from classification.yoga_pose_target_data import TARGET_ANGLES
from classification.yoga_pose_target_data import joint_configs
from .Language_Model import get_llama_feedback

# Load Grok api key here ??

# TODO:: Make this function capable of being async
def generate_pose_feedback(
        angles_dict: Dict[str, Any],
        action_type: str
):
    joint_angle_values = angles_dict
    def remove_prefix(joint_angle_values):
        jav = {}
        for k, v in joint_angle_values.items():
            jav[k] = {ik[:-3]: iv for ik, iv in v.items()}
        return jav
    joint_angle_values = remove_prefix(joint_angle_values)
    detailed_jav = {}
    for k, v in joint_angle_values[0].items():
        reference_joints = joint_configs[k]['joint_names']
        detailed_jav[k] = (reference_joints, v)
        pass
    prompt,fault_found = gen_prompt(detailed_jav,
                          TARGET_ANGLES[action_type],
                          action_type,
                          joint_configs)
    print(prompt)
    model_feedback = get_llama_feedback(prompt, temperature=0.1)
    return model_feedback, fault_found
    pass


