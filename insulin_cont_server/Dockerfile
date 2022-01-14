FROM python:3.7.12-slim-buster
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 cmake libopenmpi-dev python3-dev zlib1g-dev -y
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD python -u server.py