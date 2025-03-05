This is a python server built with fastapi webserver framework.

It serves some web endpoints for processing videos, as such, it will require ffmpeg to be installed and available to path when the server runs.

The server is by default served at `0.0.0.0:8080`.

Currently no actual frontend is built to access the api, but rather tested using fastapi's own mechanism, which is available at `/docs`.

Some features as video frame sampling, resolution downsampling is provided that can be used from docs.

Currently, the backend also provides the video playing mechanism using opencv, so to play video, one has to be able to use GUI built opencv-python.

Variety of services produce video data as result, which is then stored in a `answer` like variable internally, so later in other services, if input file is omitted, then the previous file generated at `answer` is used automatically.


# Things to do before building and running docker

+ Pull the repository
+ Create a '.env' file at the root with groq api key as
`GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
+ Build the docker image as
`docker build -t project-backend-docker .`
This will create a Docker image with the tag `project-backend-docker`.
+ Then run the container and expose the port, which is by default 8080,
`docker run -p 8080:8080 project-backend-docker`
+ Run the frontend currently from `https://bipul018.github.io/major-project-react-app/` and then go to 'Demo'



# To run
+ You need to have set the backend api url properly in frontend, (mind the slashes). By default it points to localhost:8080.
+ To show smplr-x generated mesh, you also need to run the notebook at `` (requires internet access) and paste the generated temporary gradio mesh link back onto the field in the demo.
+ To run on webcam, click the button to switch to webcam (and might need to start play)
+ To run on video, click 'upload video'
+ Then click start streaming/ stop streaming to begin and stop the streaming mechanism
