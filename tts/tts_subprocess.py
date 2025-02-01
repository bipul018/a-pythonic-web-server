# tts_service.py

# This file is to only be launched from the root folder using the process piping mechanism

from text_to_speech import text_to_speech
import sys
import os
import json
# from your_tts_module import initialize_tts, generate_speech

# Create FIFO paths
INPUT_PIPE = "/tmp/tts_input.fifo"
OUTPUT_PIPE = "/tmp/tts_output.fifo"

def main():
    # Create FIFOs if they don't exist
    if not os.path.exists(INPUT_PIPE):
        os.mkfifo(INPUT_PIPE)
    if not os.path.exists(OUTPUT_PIPE):
        os.mkfifo(OUTPUT_PIPE)

    # tts_engine = initialize_tts()
    print(f"TTS Service started, input pipe = `{INPUT_PIPE}`, output pipe = `{OUTPUT_PIPE}`. Waiting for requests...")
    
    # Reply with something so that caller knows we've initialized
    with open(OUTPUT_PIPE, 'w') as outfile:
        json.dump({"status": "initialized"}, outfile)
        pass
    print("Replied with a initialized message")

    while True:
        # Read from input FIFO
        with open(INPUT_PIPE, 'r') as infile:
            request = json.loads(infile.read())
            filename = request['filename']
            text = request['text']
            
            print("Inside service, Generating speech ... ")
            # Generate speech
            # generate_speech(filename, text, tts_engine)
            #with open(filename, 'w') as f:
            text_to_speech(text, file_or_name=filename)
            # f.write(f"Hello file {filename}, we've written onto ye `{text}`")
            #pass
            
            print("....Inside service, generated speech")
            # Write completion status
            with open(OUTPUT_PIPE, 'w') as outfile:
                json.dump({"status": "done", "filename": filename}, outfile)

if __name__ == "__main__":
    if sys.argv[1]:
        INPUT_PIPE = sys.argv[1]
        pass
    if sys.argv[2]:
        OUTPUT_PIPE = sys.argv[2] 
        pass
    main()
