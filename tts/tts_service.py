import os
import subprocess
import threading
import json
import time
from tempfile import NamedTemporaryFile as tfile

# Stole the code from tts file, 
import os
import sys
from importlib.util import spec_from_file_location, module_from_spec
import importlib

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

import wave
class TTS_Service_Piper:
    def __init__(self):
        tts_model = os.path.join("./tts/piper_models/", "en_US-hfc_female-medium.onnx")
        self.voice = importlib.import_module("piper.voice").PiperVoice.load(tts_model)
        self.lock = threading.Lock()
        pass
    
    def __enter__(self):
        return self
    def close(self):
        # self.process.terminate()
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        pass
    def generate(self, filename, text):
        # Ensure that only one tts action happens at a time
        # TODO:: If this dont work use lock in another type of tts service too!!!
        self.lock.acquire()
        with wave.open(filename, 'wb') as wav_file:
            self.voice.synthesize(text, wav_file)
            pass
        self.lock.release()
        pass



class TTS_Service_This_Process:
    def __init__(self):
        # tts_path = os.path.join("./tts", "text_to_speech.py")
        self.tts_module = import_module_from_path("tts", "./tts/text_to_speech.py")
        self.lock = threading.Lock()
        pass
    
    def __enter__(self):
        return self
    def close(self):
        self.process.terminate()
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        pass
    def generate(self, filename, text):
        # Ensure that only one tts action happens at a time
        # TODO:: If this dont work use lock in another type of tts service too!!!
        self.lock.acquire()
        self.tts_module.text_to_speech(text, file_or_name=filename)
        self.lock.release()
        pass


class TTS_Service_Another_Process:
    def __init__(self):
        self.INPUT_PIPE = "/tmp/tts_input.fifo"
        self.OUTPUT_PIPE = "/tmp/tts_output.fifo"
        with tfile(mode='wb', suffix='.fifo') as fi:
            with tfile(mode='wb', suffix='.fifo') as fo:
                self.INPUT_PIPE = fi.name
                self.OUTPUT_PIPE = fo.name
                pass
            pass
        self.process=subprocess.Popen(["python", "tts/tts_subprocess.py",
                                       self.INPUT_PIPE, self.OUTPUT_PIPE])
        # Ensure that it has inititlzied
        breakit = False
        while not breakit:
            try:
                with open(self.OUTPUT_PIPE, 'r') as outfile:
                    response = outfile.read()
                    pass
                breakit = True
                pass
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)  # Wait briefly before retrying
                pass
        pass
    
    def __enter__(self):
        return self
    def close(self):
        self.process.terminate()
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        pass
    def generate(self, filename, text):
        # Create request object
        request = {"filename": filename, "text": text}
    
        # Write to input FIFO
        with open(self.INPUT_PIPE, 'w') as infile:
            json.dump(request, infile)
            pass
        #print("Waiting for answer...")
        # Wait for and read completion status
        while True:
            try:
                with open(self.OUTPUT_PIPE, 'r') as outfile:
                    response = json.load(outfile)
                    if response['filename'] == filename:
                        return response
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)  # Wait briefly before retrying
        pass
