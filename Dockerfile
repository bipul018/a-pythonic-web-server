# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install apt-get items
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg espeak-ng

# Copy the original requirements.txt
COPY requirements.txt .

# Replace 'opencv-python' with 'opencv-python-headless', then install

# Use sed to modify requirements.txt and pipe it directly into pip install
RUN sed 's/^opencv-python/opencv-python-headless/' requirements.txt | pip install --no-cache-dir -r /dev/stdin
# RUN sed 's/^opencv-python/opencv-python-headless/' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 8080

# Run the FastAPI app using uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "serv.py"]
