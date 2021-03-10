from spleeter.audio.adapter import AudioAdapter

import numpy as np
import pandas as pd
from tqdm import tqdm

import gc

class FileLoader():

    def __init__(self) -> None:
        self._audio_loader = AudioAdapter.default()

    def load(self, df):
        collection = []
        for (_, file, label) in tqdm(df.itertuples(), total=df.shape[0], desc='Loading Files'):
            try:
                audio = self._audio_loader.load(file, sample_rate=44100, duration=30.0)
                if audio[0].shape[1] == 1:
                    data = [ label , [np.repeat(a=audio[0], repeats=2, axis=1)] ]
                else:  
                    data = [ label , [audio[0]] ]
                collection.append(data)
            except Exception as e :
                print(e)

        gc.collect()
        return collection
