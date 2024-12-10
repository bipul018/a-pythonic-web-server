import asyncio
import base64
import io
import enum
from typing import Dict, Any, List, Callable, Optional, Union, Type
from contextlib import contextmanager
import traceback

import fastapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
import uvicorn
# import bson
import json as bson

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

# FastAPI Application
app = FastAPI()
connection_manager = ConnectionManager()

# Service Instances
import tryvds
import tempfile

@app.get("/", response_class=HTMLResponse)
async def root():
    with open('test1.html', 'r') as file:
        return file.read()

@app.get("/jpt", response_class=HTMLResponse)
async def jpt():
    return """
    <html>
    <head>
    <title> Hello World </title>
    </head>
    <body> Hello world, this is the jpt file </body>
    </html>
    """

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

class VideoArgAsFile_:
    #def __init__(self, videoUpload: Optional[fastapi.UploadFile]):
    #    if (videoUpload is None) and (latest_video_bytes is None):
    #        raise "No Video File available to process"
    #    if videoUpload is None:
    #        data = latest_video_bytes
    #    else:
    #        data = await videoUpload.read()
    #    self.file = tempfile.NamedTemporaryFile(mode='wb')
    #    print(f"Actual file is {videoUpload.filename}, file size is {len(data)}, temp file is {self.file.name}")
    #    self.file.write(data)
    #    self.name = self.file.name
    def __init__(self, data):
        self.file = tempfile.NamedTemporaryFile(mode='wb')
        print(f"Uploaded file size is {len(data)}, temp file is {self.file.name}")
        self.file.write(data)
        self.name = self.file.name
    def __enter__(self):
        return self
    def close(self):
        self.file.close()
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
async def VideoArgAsFile(videoUpload: Optional[fastapi.UploadFile]):
    if (videoUpload is None) and (latest_video_bytes is None):
        raise Exception("No Video File available to process")
    if videoUpload is None:
        data = latest_video_bytes
    else:
        data = await videoUpload.read()
    return VideoArgAsFile_(data)
        

# @contextmanager
# async def get_video_arg_as_file(videoUpload: Optional[fastapi.UploadFile]):
#     if (videoUpload is None) and (latest_video_bytes is None):
#         raise "No Video File available to process"
    
#     if videoUpload is None:
#         data = latest_video_bytes
#     else:
#         data = await videoUpload.read()
#     with tempfile.NamedTemporaryFile(mode='wb') as infile:
#         print(f"Actual file is {videoUpload.filename}, file size is {len(data)}, temp file is {infile.name}")
#         infile.write(data)
#         yield infile.name

class TaskItem(BaseModel):
    name: str
    fxn: Callable[..., Any]

tasks: List[TaskItem] = []    
    
def register_task(name: str):
    def decorator(func: Callable[..., Any]):
        global tasks
        endpoint = f"/task/{name}"
        tsk = TaskItem(name=name, fxn=func)
        tasks.append(tsk)
        return app.post(endpoint)(func)
    return decorator

# Here, add all the services offered
@register_task("play_video")
async def play_video_task(videoUpload: Optional[fastapi.UploadFile]=None, frames: Optional[int]=None, fps: Optional[int]=None):
    try:
        with await VideoArgAsFile(videoUpload) as infile:
            tryvds.draw_video(infile.name, fixed_frames=frames, fixed_fps=fps)
    except Exception as e:
        print("Exception:", repr(e))
        print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
        return {'status': f'Error {e}'}
    return {'status' : f'Success'}

@register_task("downsample_it")
async def downsample_video_task(video: Optional[fastapi.UploadFile]=None, factor: int = 2):
    try:
        with await VideoArgAsFile(video) as infile:
            outdat = tryvds.downsample_it(infile.name, factor)
            update_video_bytes(outdat)
            print(f"The result of downsampling of size {len(outdat)}")
            return {'status' : 'Success',
                    'value' : f'Downsampled file size is {len(outdat)}'}
            #return {'status' : f'Success',
                    #'value' : outdat}
                    #'value' : Response(content=outdat, media_type='video/mp4')}
            #return Response(content=outdat, media_type='video/mp4')
    except Exception as e:
        print("Exception:", repr(e))
        print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
        return {'status': 'Error',
                'value' : f"{e}"}

@register_task("select_at_fps")
async def select_frames_at_fps_task(video: Optional[fastapi.UploadFile]=None, fps: int = 1):
    try:
        with await VideoArgAsFile(video) as infile:
            outdat = tryvds.sample_at_fps(infile.name, fps)
            update_video_bytes(outdat)
            return {'status' : 'Success',
                    'value' : f'FPS resampled file size is {len(outdat)}'}
    except Exception as e:
        print("Exception:", repr(e))
        print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
        return {'status': 'Error',
                'value' : f"{e}"}

@register_task("select_frames")
async def select_fixed_frames_task(video: Optional[fastapi.UploadFile]=None, frames: int = 21):
    try:
        with await VideoArgAsFile(video) as infile:
            outdat = tryvds.sample_n_frames(infile.name, frames)
            update_video_bytes(outdat)
            return {'status' : 'Success',
                    'value' : f'After selecting frames, file size is {len(outdat)}'}
    except Exception as e:
        print("Exception:", repr(e))
        print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
        return {'status': 'Error',
                'value' : f"{e}"}

@register_task("query_info")
async def query_video_info_task(video: Optional[fastapi.UploadFile]=None):
    try:
        with await VideoArgAsFile(video) as infile:
            anses = tryvds.query_info(infile.name)
            return {'status' : 'Success',
                    'value' : anses}
    except Exception as e:
        print("Exception:", repr(e))
        print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
        return {'status': 'Error',
                'value' : f"{e}"}

@register_task("clear_last_video")
async def clear_last_video_saved_task():
    prev_value = latest_video_bytes
    update_video_bytes(None)
    return {'status' : 'Success',
            'value' : f'Cleared video file size was {len(prev_value)}'}

@register_task("save_video")
async def save_video_for_later_task(video: fastapi.UploadFile):
    try:
        data = await video.read()
        update_video_bytes(data)
        return {'status' : 'Success',
                'value' : f'Saved video size is {len(data)}'}
    except Exception as e:
        print("Exception:", repr(e))
        print("Stacktrace of exception:\n", traceback.print_tb(e.__traceback__))
        return {'status': 'Error',
                'value' : f"{e}"}
    
@register_task("restore_video")
async def restore_previous_video_task():
    restore_video_bytes()
    old_bytes = latest_video_bytes
    return {'status' : 'Success',
            'value' : f'Cleared video file size was {len(old_bytes)}'}

            
    

#@app.websocket("/ws")
#async def websocket_endpoint(websocket: WebSocket):
#    await connection_manager.connect(websocket)
#    try:
#        while True:
#            # Receive BSON-encoded message
#            #bson_message = await websocket.receive_bytes()
#            bson_message = await websocket.receive_text()
#            message = bson.loads(bson_message)
#
#            # Route to appropriate service based on message type
#            service_type = message.get('service', 'unknown')
#            
#            result = None
#            if service_type == 'image_processing':
#                result = await image_service.process_image(message.get('image'))
#            
#            elif service_type == 'data_analytics':
#                result = await analytics_service.analyze_data(message.get('data'))
#            
#            elif service_type == 'ml_training':
#                result = await training_service.start_training(message.get('config'))
#            
#            else:
#                result = {'error': 'Unknown service'}
#
#            # Send response back as BSON
#            response = bson.dumps({
#                'service': service_type,
#                'result': result
#            })
#            await websocket.send_bytes(response)
#
#    except WebSocketDisconnect:
#        connection_manager.disconnect(websocket)

# Start server
if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8080)
    uvicorn.run("serv:app", host="0.0.0.0", port=8080, reload=True)
    
