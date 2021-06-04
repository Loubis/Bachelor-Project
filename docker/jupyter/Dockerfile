FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

RUN apt-get update && apt-get -y install ffmpeg graphviz

RUN pip install --upgrade pip

RUN pip install IPython spleeter nussl tqdm pydot 