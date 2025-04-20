from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def get_llama_feedback(
    user_content, 
    system_content="You are a helpful yoga coach providing precise, constructive feedback.",
    model="llama-3.1-8b-instant", 
    temperature=0, 
    max_completion_tokens=1024, 
    top_p=1
):
    """
    Generate feedback using Groq's Llama API.
    
    Args:
        user_content (str): The main content/prompt to send to the model
        system_content (str, optional): System message to set model context. Defaults to fitness coach prompt.
        model (str, optional): Model to use. Defaults to "llama-3.1-8b-instant".
        temperature (float, optional): Sampling temperature. Defaults to 1.
        max_completion_tokens (int, optional): Maximum tokens in response. Defaults to 1024.
        top_p (float, optional): Nucleus sampling parameter. Defaults to 1.
    
    Returns:
        str: Generated model feedback
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))  

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": system_content
            },
            {
                "role": "user", 
                "content": user_content
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=True,
        stop=None,
    )

    model_feedback = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        print(content, end="")
        model_feedback += content
    
    return model_feedback


import requests

def get_local_language_model_feedback(
    user_content, 
    system_content="You are a helpful fitness coach providing precise, constructive feedback.",
    model="local-model", 
    temperature=0, 
    max_tokens=1024, 
    top_p=1,
    base_url="http://localhost:1234/v1"
):
    """
    Generate feedback using LM Studio's local server.
    
    Args:
        user_content (str): The main content/prompt to send to the model
        system_content (str, optional): System message to set model context. 
        model (str, optional): Model identifier. Defaults to "local-model".
        temperature (float, optional): Sampling temperature. Defaults to 1.
        max_tokens (int, optional): Maximum tokens in response. Defaults to 1024.
        top_p (float, optional): Nucleus sampling parameter. Defaults to 1.
        base_url (str, optional): Base URL for LM Studio server. Defaults to local endpoint.
    
    Returns:
        str: Generated model feedback
    """
    try:
        # Prepare the payload for the API request
        payload = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "model": model
        }

        # Send POST request to LM Studio server
        response = requests.post(
            f"{base_url}/chat/completions", 
            json=payload
        )

        # Check if the request was successful
        response.raise_for_status()

        # Extract and return the model's response
        model_response = response.json()
        model_feedback = model_response['choices'][0]['message']['content'].strip()

        # Optional: Print the feedback as it's generated
        print(model_feedback)
        
        return model_feedback

    except requests.RequestException as e:
        print(f"Error communicating with LM Studio server: {e}")
        return ""


import torch
from typing import Dict, Tuple, Any, Union
from classification.yoga_pose_target_data import ACTION_JOINT_MAPPING

def format_angle_data(angle_data: Dict[str, Union[Tuple, torch.Tensor]]) -> Dict[str, float]:
    """
    Convert the raw angle data into a clean dictionary of float values
    """
    formatted_data = {}
    for joint, data in angle_data.items():
        # Check if the data is a tuple (joints, angle) or just an angle
        if isinstance(data, tuple):
            angle = data[1]
        else:
            angle = data
        
        # Convert tensor to float
        if isinstance(angle, torch.Tensor):
            formatted_data[joint] = float(angle)
        else:
            formatted_data[joint] = float(angle)
    
    return formatted_data

def generate_pose_feedback_prompt(
    user_angles: Dict[str, Any],
    target_angles: Dict[str, Any],
    action_type: str,
    joint_configs: Dict[str, Dict[str, Any]]
) -> str:
    """
    Generate a structured prompt for pose feedback
    
    Parameters:
    - user_angles: Dictionary of current user joint angles
    - target_angles: Dictionary of target joint angles
    - action_type: Type of pose/action being performed
    - joint_configs: Configuration dictionary containing joint relationships
    
    Returns:
    - Formatted prompt string
    """
    # Format the angle data
    user_angles_clean = format_angle_data(user_angles)
    target_angles_clean = format_angle_data(target_angles)
    
    # Get relevant joints for this action
    relevant_joints = ACTION_JOINT_MAPPING.get(action_type, [])
    prompt_parts = [
        f"Action: {action_type} Pose\n",
        "Current Joint Analysis:\n"
    ]
    
    # Add joint-specific information
    for joint in relevant_joints:
        if joint in user_angles_clean and joint in target_angles_clean:
            current_angle = user_angles_clean[joint]
            target_angle = target_angles_clean[joint]
            # print(current_angle)
            
            # Get involved body parts from joint_configs
            involved_parts = joint_configs[joint]['joint_names']
            
            prompt_parts.append(
                f"- {joint.replace('_', ' ').title()}:\n"
                f"  Current: {current_angle:.1f}°\n"
                f"  Target: {target_angle:.1f}°\n"
                f"  Involved parts: {', '.join(involved_parts)}\n"
            )
    
    prompt_parts.extend([
        "\nContext:",
        f"- This is a {action_type.replace('_', ' ')} yoga-pose.",
        "- Focus on angles representing proper body alignment.",
        "Task: Respond with the feedback in 5-10 words only like a yoga instructor.",
        "Avoid any extra explanations or numbers.",
        "Only focus on most significant flaw and give output like the user is listening to the feedback while doing the pose."
    ])
        # "Keep it short, direct, and actionable, like a yoga instructor would.",
    # prompt_parts.extend([
    #     "\nContext:",
    #     f"- This is a {action_type.replace('_', ' ')} position.",
    #     "- Focus on angles representing proper body alignment.",
    #     "Task: Respond ONLY with the coach's feedback in 5-10 words. Avoid any extra explanations or numbers.",
    #     "Keep it short, direct, and actionable, like a yoga instructor would.",
    #     "Only focus on most significant flaw and give output like the user is listening while doing the pose."
    # ])
    return "\n".join(prompt_parts)

from classification.yoga_pose_target_data import PRIORITY_JOINT_MAPPING, POSE_BREATHING_HOLD_STATE
def generate_short_prompt(
    user_angles: Dict[str, Any],
    target_angles: Dict[str, Any],
    action_type: str,
    joint_configs: Dict[str, Dict[str, Any]]
) -> Tuple[str, bool]:
    """
    Generate a structured prompt for pose feedback
    
    Parameters:
    - user_angles: Dictionary of current user joint angles
    - target_angles: Dictionary of target joint angles
    - action_type: Type of pose/action being performed
    - joint_configs: Configuration dictionary containing joint relationships
    
    Returns:
    - Formatted prompt string and bool is fault was found or not
    """
    # Format the angle data
    user_angles_clean = format_angle_data(user_angles)
    target_angles_clean = format_angle_data(target_angles)
    
    # Get relevant joints for this action
    relevant_joints = PRIORITY_JOINT_MAPPING.get(action_type, [])
    prompt_parts = [
        f"Action: {action_type} Pose\n",
        "Current Joint Analysis:\n"
    ]
    
    # Add joint-specific information, focusing on the first significant deviation based on priority
    ANGLE_TOLERANCE = 30  # Threshold for significant angle difference
    FAULT_FOUND = False

    for joint in relevant_joints:
        current_angle = user_angles_clean.get(joint)
        target_angle = target_angles_clean.get(joint)

        if current_angle is not None and target_angle is not None:
            deviation = target_angle - current_angle
            if abs(deviation) > ANGLE_TOLERANCE:
                FAULT_FOUND = True
                involved_parts = joint_configs[joint]['joint_names']
                joint_name_formatted = joint.replace('_', ' ').title()
                direction = "increase" if deviation > 0 else "decrease"
                prompt_parts.append(
                    f"- {joint_name_formatted}:\n"
                    f"  Current: {current_angle:.1f}°\n"
                    f"  Target: {target_angle:.1f}°\n"
                    f"  Involved parts: {', '.join(involved_parts)}\n"
                )
                break
    

    if FAULT_FOUND: 
        prompt_parts.extend([
            "\nContext:",
            f"- This is a {action_type.replace('_', ' ')} yoga-pose.",
            "- Focus on angles representing proper body alignment.",
            "Task: Respond with the feedback in 5-10 words only like a yoga instructor.",
            "Avoid any extra explanations or numbers.",
            "Give output like the user is listening to the feedback while doing the pose."
        ])

        return "\n".join(prompt_parts), FAULT_FOUND
    else:
        # No significant fault found, provide positive reinforcement
        breathing_type = POSE_BREATHING_HOLD_STATE.get(action_type, "steady breaths") # Use default if not found

        prompt_parts = [
            f"Action: {action_type} Pose\n",
            "Current Joint Analysis: All checked joints are within acceptable tolerance.\n",
            "Context:",
            f"- The user is holding the {action_type.replace('_', ' ')} yoga pose correctly.",
            f"- Recommended breathing for this pose: '{breathing_type}'.",
            "\nTask:",
            "Provide brief (5-10 words) encouraging feedback like a yoga instructor.",
            "Confirm the pose is correct and incorporate the recommended breathing advice.",
            "Keep it concise, positive, and encouraging.",
        ]
        return "\n".join(prompt_parts), FAULT_FOUND


