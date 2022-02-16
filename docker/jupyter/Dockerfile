FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

RUN apt-get update && apt-get -y install libsndfile1 ffmpeg graphviz

RUN pip install --upgrade pip

RUN pip install -U ipykernel

RUN pip install soundfile IPython spleeter tqdm pydot 