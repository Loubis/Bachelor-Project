FROM tensorflow/tensorflow:2.5.0-gpu

RUN apt-get update && apt-get -y install ffmpeg graphviz libsndfile1-dev

RUN pip install --upgrade pip

RUN pip install spleeter==2.3.0 tqdm pydot