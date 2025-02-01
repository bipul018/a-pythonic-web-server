import os
import subprocess
import json
import time
from tempfile import NamedTemporaryFile as tfile

class TTS_Service:
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
