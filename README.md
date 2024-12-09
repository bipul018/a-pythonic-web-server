This is a python server built with fastapi webserver framework.

It serves some web endpoints for processing videos, as such, it will require ffmpeg to be installed and available to path when the server runs.

The server is by default served at `0.0.0.0:8080`.

Currently no actual frontend is built to access the api, but rather tested using fastapi's own mechanism, which is available at `/docs`.

Some features as video frame sampling, resolution downsampling is provided that can be used from docs.

Currently, the backend also provides the video playing mechanism using opencv, so to play video, one has to be able to use GUI built opencv-python.

Variety of services produce video data as result, which is then stored in a `answer` like variable internally, so later in other services, if input file is omitted, then the previous file generated at `answer` is used automatically.
