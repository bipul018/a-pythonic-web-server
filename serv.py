import asyncio
import base64
import io
import enum
from typing import Dict, Any, List, Callable, Optional, Union, Type
from contextlib import contextmanager
import traceback

import fastapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json


# FastAPI Application
app = FastAPI()

# Add the frontend port service as a safe CORS thing
#  TODO:: Find out if when served through some public name, still it works or not??
origins = [
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Instances
import video_tasks
import tempfile

@app.get("/", response_class=HTMLResponse)
async def root():
    with open('test1.html', 'r') as file:
        return file.read()

# Pseudo session mechanism
# stores the latest video object returned by any one of the video returning fxns
# if no new video is provided, this will be played

previous_video_bytes =  None
latest_video_bytes = None
def update_video_bytes(new_bytes):
    global latest_video_bytes
    global previous_video_bytes
    previous_video_bytes = latest_video_bytes
    latest_video_bytes = new_bytes

def restore_video_bytes():
    global latest_video_bytes
    latest_video_bytes = previous_video_bytes

async def VideoArgAsFile(videoUpload: Optional[fastapi.UploadFile]):
    if (videoUpload is None) and (latest_video_bytes is None):
        raise Exception("No Video File available to process")
    if videoUpload is None:
        data = latest_video_bytes
    else:
        data = await videoUpload.read()
    return io.BytesIO(data)

class TaskItem(BaseModel):
    name: str
    fxn: Callable[..., Any]

curr_task_list: List[TaskItem] = []    
log_traceback_on_error = True
import functools
def register_task(name: str, **kwargs):
    def decorator(func: Callable[..., Any]):
        global curr_task_list
        endpoint = f"/task/{name}"
        tsk = TaskItem(name=name, fxn=func)
        curr_task_list.append(tsk)
        @functools.wraps(func)
        async def wrapped_func(*args, **kwargs):
            try:
                print(f"The args are {args} and kwargs are {kwargs} for the incoming request.")
                ans = await func(*args, **kwargs)
                if ans is None:
                    return {'status': 'Success'}
                return {'status' : 'Success',
                        'value' : ans}
            except Exception as e:
                if log_traceback_on_error:
                    print("Exception:", repr(e))
                    print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
                return {'status': 'Error',
                        'value' : f"{e}"}
        return app.post(endpoint, **kwargs)(wrapped_func)
    return decorator

# Here, add all the services offered
@register_task("play_video")
async def play_video_task(videoUpload: Optional[fastapi.UploadFile]=None, frames: Optional[int]=None, fps: Optional[int]=None):
    with await VideoArgAsFile(videoUpload) as infile:
        video_tasks.draw_video(infile, fixed_frames=frames, fixed_fps=fps)

@register_task("downsample_it")
async def downsample_video_task(video: Optional[fastapi.UploadFile]=None, factor: int = 2):
    with await VideoArgAsFile(video) as infile:
        outdat = video_tasks.downsample_it(infile, factor)
        update_video_bytes(outdat)
        print(f"The result of downsampling of size {len(outdat)}")
        return f'Downsampled file size is {len(outdat)}'

@register_task("draw_landmarks")
async def draw_landmarks_on_video_task(video: Optional[fastapi.UploadFile]=None):
    with await VideoArgAsFile(video) as infile:
        outdat = video_tasks.draw_landmarks_on_video(infile)
        update_video_bytes(outdat)
        print(f"The result of drawing landmarks was of size {len(outdat)}")
        return f'file size is {len(outdat)}'

@register_task("stsae_gcn_prediction")
async def predict_using_stsae_gcn_on_video_task(video: Optional[fastapi.UploadFile]=None):
    with await VideoArgAsFile(video) as infile:
        return video_tasks.infer_stsae_prediction_on_video(infile)
    
@register_task("select_at_fps")
async def select_frames_at_fps_task(video: Optional[fastapi.UploadFile]=None, fps: int = 1):
    with await VideoArgAsFile(video) as infile:
        outdat = video_tasks.sample_at_fps(infile, fps)
        update_video_bytes(outdat)
        return f'FPS resampled file size is {len(outdat)}'

@register_task("select_frames")
async def select_fixed_frames_task(video: Optional[fastapi.UploadFile]=None, frames: int = 21):
    with await VideoArgAsFile(video) as infile:
        outdat = video_tasks.sample_n_frames(infile, frames)
        update_video_bytes(outdat)
        return f'After selecting frames, file size is {len(outdat)}'

@register_task("query_info")
async def query_video_info_task(video: Optional[fastapi.UploadFile]=None):
    with await VideoArgAsFile(video) as infile:
        anses = video_tasks.query_info(infile)
        return anses

@register_task("clear_last_video")
async def clear_last_video_saved_task():
    prev_value = latest_video_bytes
    update_video_bytes(None)
    return f'Cleared video file size was {len(prev_value)}'

@register_task("save_video")
async def save_video_for_later_task(video: fastapi.UploadFile):
    # TODO:: The input video might not be in mp4 format at all !!! Need to
    #        either detect that, or always just save as a mp4 format video
    data = await video.read()
    update_video_bytes(data)
    return f'Saved video size is {len(data)}'
    
@register_task("restore_video")
async def restore_previous_video_task():
    restore_video_bytes()
    old_bytes = latest_video_bytes
    return f'Cleared video file size was {len(old_bytes)}'

@register_task("get_video")
async def get_last_video_as_bytes():
    if latest_video_bytes is None:
        raise Exception("Cannot return video when none has been saved")
    return {"type" : "video/mp4",
            "bytes" : base64.b64encode(latest_video_bytes).decode('utf-8')}


import stream.service
from stream.service import ConnectionHandler
stream.service.update_video_bytes = update_video_bytes

@app.websocket("/wsprocess_frame")
async def websocket_process_frame(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    decoder = json.JSONDecoder()
    service_provider = ConnectionHandler()
    def decode_msg(msg):
        #print(f"Decoding message ... ")
        try:
            utf8_msg = msg.decode('utf-8')
            bin_data = b''
        except UnicodeDecodeError as e:
            bin_data = msg[e.start:]
            utf8_msg = msg[:e.start].decode('utf-8')
        obj, inx = decoder.raw_decode(utf8_msg)
        trailer = utf8_msg[inx:].encode()
        return obj, trailer+bin_data
    try:
        while True:
            #print(f"Partitioning metadata ... ")
            metadata, data = decode_msg(await websocket.receive_bytes())
            #print(f"Dumped metadata : {metadata}")
            service_provider.new_data(metadata, data) # async fxn
            # Check if any pending replies are there, if so, send them one by one
            for reply in service_provider.pop_replies():
                reply = json.dumps(reply)
                print(f"replying with {reply}....")
                await websocket.send_text(reply) # async fxn
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("Exception during websocket processing:", repr(e))
        print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
    finally:
        # TODO:: Find out if this interferes with the async fxns not awaited above
        service_provider.on_close()
        try:
            # TODO:: Find out why this throws exception
            await websocket.close()
        except Exception:
            pass
        print("WebSocket connection closed.")

# Start server
if __name__ == "__main__":
    uvicorn.run("serv:app", host="0.0.0.0", port=8080, reload=True)
    
