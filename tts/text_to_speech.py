import torch
import os
import sys
import platform
import glob
import warnings
# from huggingface_hub import hf_hub_download, list_repo_files
# import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from pathlib import Path

# Filter out specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# __all__ = ['list_available_voices', 'build_model', 'load_voice', 'generate_speech', 'load_and_validate_voice']
import os

# Get current working directory
current_dir = os.path.join(os.getcwd(), 'tts')
tts_model_path = os.path.join(current_dir, 'tts_model')
voices_path = os.path.join(tts_model_path, 'voices')
model_path = os.path.join(tts_model_path, 'kokoro-v0_19.pth')
# Ensure model_path exists
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    model_path = None
    

def get_platform_paths():
    """Get platform-specific paths for espeak-ng"""
    system = platform.system().lower()
    
    # TODO:: Also allow using environment variable to be set

    if system == "windows":
        lib_path = os.path.join(os.getenv("ProgramFiles"), "eSpeak NG", "libespeak-ng.dll")
        data_path = os.path.join(os.getenv("ProgramFiles"), "eSpeak NG", "espeak-ng-data")
    
    elif system == "darwin":  # macOS
        lib_path = "/opt/homebrew/lib/libespeak-ng.dylib"
        brew_data = "/opt/homebrew/share/espeak-ng-data"
        sys_lib = "/usr/local/lib/libespeak-ng.dylib"
        sys_data = "/usr/local/share/espeak-ng-data"
        lib_path = lib_path if os.path.exists(lib_path) else sys_lib
        data_path = brew_data if os.path.exists(brew_data) else sys_data
    
    else:  # Linux
        data_paths = [
            os.environ.get('LIBSPEAK_NG_DATA', ''),
            "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
        ]
        lib_paths = [
            os.environ.get('LIBSPEAK_NG_LIB', '') + "/libespeak-ng.so.1",
            os.environ.get('LIBSPEAK_NG_LIB', '') + "/libespeak-ng.so",
            "/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
            "/usr/lib/libespeak-ng.so",
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/aarch64-linux-gnu/libespeak-ng.so",
            "/usr/lib64/libespeak-ng.so"
        ]
        
        lib_path = None
        for path in lib_paths:
            if os.path.exists(path):
                lib_path = path
                break
        
        if lib_path is None:
            lib_path = lib_paths[0]  # Default for error message
        
        data_path = None
        for path in data_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            data_path = data_paths[0]  # Default for error message
    
    return lib_path, data_path

def setup_espeak():
    """Set up espeak library paths for phonemizer."""
    try:
        lib_path, data_path = get_platform_paths()
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"espeak-ng library not found at {lib_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"espeak-ng data not found at {data_path}")
            
        EspeakWrapper.set_library(lib_path)
        EspeakWrapper.data_path = data_path
        
        # Configure phonemizer for UTF-8
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
        os.environ["PHONEMIZER_ESPEAK_PATH"] = data_path
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        print("espeak-ng library paths set up successfully")
        
    except Exception as e:
        print(f"Error setting up espeak: {e}")
        print("\nPlease ensure espeak-ng is installed:")
        print("- Windows: Download from https://github.com/espeak-ng/espeak-ng/releases")
        print("- macOS: brew install espeak-ng")
        print("- Linux: sudo apt install espeak-ng")
        raise e

import os
import sys
from importlib.util import spec_from_file_location, module_from_spec

def import_module_from_path(module_name, module_path):
    """Import a module from file path."""
    try:
        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing module {module_name}: {e}")
        raise e

def build_model(local_folder, device='cpu'):
    """
    Build the Kokoro TTS model for offline usage.
    Args:
        local_folder (str): Path to the folder containing all required files.
        device (str): Device to load the model on ('cpu' or 'cuda').
    Returns:
        The Kokoro TTS model instance.
    """
    try:
        setup_espeak()
        # Path to all required files
        model_path = os.path.join(local_folder, "kokoro-v0_19.pth")
        kokoro_py = os.path.join(local_folder, "kokoro.py")
        models_py = os.path.join(local_folder, "models.py")
        istftnet_py = os.path.join(local_folder, "istftnet.py")
        plbert_py = os.path.join(local_folder, "plbert.py")
        config_json = os.path.join(local_folder, "config.json")

        # Import required modules
        print("Importing plbert module...")
        plbert_module = import_module_from_path("plbert", plbert_py)
        print("Importing istftnet module...")
        istftnet_module = import_module_from_path("istftnet", istftnet_py)
        print("Importing models module...")
        models_module = import_module_from_path("models", models_py)
        print("Importing kokoro module...")
        kokoro_module = import_module_from_path("kokoro", kokoro_py)

        # Test phonemizer (optional; ensure phonemizer is installed)
        from phonemizer import phonemize
        test_phonemes = phonemize("Hello")
        print(f"Phonemizer test successful: 'Hello' -> {test_phonemes}")

        # Build the model
        print("Building model...")
        model = models_module.build_model(model_path, device)
        print(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        raise e

def list_available_voices(voices_path = voices_path):
    try:
        # Ensure voices directory exists
        if not os.path.exists(voices_path):
            print(f"Voices directory does not exist at {voices_path}. Creating directory...")
            os.makedirs(voices_path)
            return []
        # List all files in the voices directory
        voice_files = os.listdir(voices_path)
        voices = [
            os.path.splitext(voice)[0] 
            for voice in voice_files 
            if voice.endswith('.pt') or voice.endswith('.pth')
        ]
        return voices
    except Exception as e:
        print(f"Error listing voices: {e}")
        return []


def load_voice(voice_name: str, device: str):
    """Load a specific voice tensor from the voices directory.
    
    Args:
        voice_name: Name of the voice file to load (without extension)
        device: Device to load the tensor on ('cuda' or 'cpu')
    
    Returns:
        Loaded voice tensor
    """
    voice_file_path = os.path.join(voices_path, f"{voice_name}.pt")
    
    # Check if the voice file exists
    if not os.path.exists(voice_file_path):
        voice_file_path = os.path.join(voices_path, f"{voice_name}.pth")
        
    if not os.path.exists(voice_file_path):
        raise ValueError(f"Voice file for '{voice_name}' not found in {voices_path}")
    
    # Load the voice tensor
    voice_tensor = torch.load(voice_file_path, weights_only=True, map_location=device)
    return voice_tensor


def load_and_validate_voice(voice_name: str = 'af_bella', device: str= 'cpu') -> torch.Tensor:
    """Load and validate the requested voice.
    
    Args:
        voice_name: Name of the voice to load
        device: Device to load the voice on ('cuda' or 'cpu')
        
    Returns:
        Loaded voice tensor
        
    Raises:
        ValueError: If the requested voice doesn't exist
    """
    available_voices = list_available_voices()
    if voice_name not in available_voices:
        raise ValueError(f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}")
    return load_voice(voice_name, device)

def generate_speech(model, text, voice=None,local_folder=tts_model_path, lang='a', device='cpu'):
    """Generate speech using the Kokoro model."""
    try:
        # repo_id = "hexgrad/Kokoro-82M"
        # kokoro_py = hf_hub_download(repo_id=repo_id, filename="kokoro.py")
        kokoro_py = os.path.join(local_folder, "kokoro.py")
        kokoro_module = import_module_from_path("kokoro", kokoro_py)
        
        # Generate speech
        audio, phonemes = kokoro_module.generate(model, text, voice, lang=lang)
        
        # Handle phonemes encoding
        if phonemes:
            try:
                # Convert to string if it's bytes
                if isinstance(phonemes, bytes):
                    phonemes = phonemes.decode('utf-8', errors='replace')
                # If it's a string, ensure it's valid UTF-8
                elif isinstance(phonemes, str):
                    # Replace problematic characters with their ASCII approximations
                    replacements = {
                        'É™': 'ə',
                        'ÊŠ': 'ʊ',
                        'Ê': 'ʃ',
                        'æ': 'ae'
                    }
                    for old, new in replacements.items():
                        phonemes = phonemes.replace(old, new)
                
                print(f"Debug - Processed phonemes: {repr(phonemes)}")
            except Exception as e:
                print(f"Debug - Encoding error: {str(e)}")
                # Last resort: strip to ASCII
                phonemes = ''.join(c for c in str(phonemes) if ord(c) < 128)
        
        return audio, phonemes
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None, None

import soundfile as sf
SAMPLE_RATE = 22050
DEFAULT_LANG ='a'

def build_tts_model(voice_name="af_bella",device:str = None):
    """Build and return TTS model and voice."""
    try:
        # Set up device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load model
        print("\nLoading model...")
        print(model_path)
        model = build_model(tts_model_path, device)

        # Load voice
        print("\nLoading voice...")
        # voice_name = "af_bella"  # Replace with your preferred voice name
        voice = load_and_validate_voice(voice_name, device)
        
        return model, voice
    except Exception as e:
        print(f"Error building TTS model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# If given a file-like object or filename, will write onto the file also
def generate_audio(model, voice, text:str, device:str = 'cpu', file_or_name=None): 
    """Generate and save audio from text."""
    try:
        print(f"\nGenerating speech for: '{text}'")
        audio, phonemes = generate_speech(model, text, voice, device=device)

        # Save output audio
        if audio is not None:
            # Display audio inline
            # from IPython.display import Audio, display
            # display(Audio(audio, rate=SAMPLE_RATE))
            
            # Optional: save to file
            # output_path = "output.wav"
            if file_or_name is not None:
                if len(audio.shape) == 1:
                    channels = 1
                    pass
                else:
                    assert 2 == len(audio.shape)
                    channels = audio.shape[1]
                    pass
                args = {'file': file_or_name, 'mode':'w',
                        'samplerate':SAMPLE_RATE,
                        'channels':channels}
                if not(isinstance(file_or_name, str) or hasattr(file_or_name, 'name')):
                    # TODO:: This is broken for some fking reason
                    args['format'] = 'wav'
                    pass
                print(f"The arguments were : {args}")
                with sf.SoundFile(**args) as f:
                    f.write(audio)
                print(f"\nAudio saved.")
                pass

            
            if phonemes:
                print(f"Generated phonemes: {phonemes}")
            
            return audio
        else:
            print("Error: Failed to generate audio")
            return None
    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        return None

# Build the TTS model with a preferred voice once
preferred_voice = "af_bella"  # Set your preferred voice here
model, voice = build_tts_model(voice_name=preferred_voice)

def text_to_speech(text, model = model, voice= voice, file_or_name='output.wav'):
    """Convert text to speech using the pre-built model and voice."""
    return generate_audio(model, voice, text, file_or_name=file_or_name)
