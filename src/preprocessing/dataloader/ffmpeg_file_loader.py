from math import fabs
import warnings
from spleeter.audio.adapter import AudioAdapter

import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

import gc

class FileLoader():
    def __init__(self) -> None:
        self._audio_adapter = AudioAdapter.default()

    def load(self, df):
        collection = []
        for (_, file, label) in tqdm(df.itertuples(), total=df.shape[0]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio, sr = librosa.load(file, mono=False, sr=44100, duration=30.0)

                # If loaded audio is only mono duplicate channel
                if audio.shape[0] != 2:
                    audio = np.array([
                        audio,
                        audio
                    ])

                # If audio is too short add padding
                if audio.shape[1] < 30*44100:
                    audio = np.array([
                        np.pad(audio[0], (0, 30*44100 - audio.shape[1]), mode='constant', constant_values=0),
                        np.pad(audio[1], (0, 30*44100 - audio.shape[1]), mode='constant', constant_values=0)
                    ])
                elif audio.shape[1] > 30*44100:
                    # TODO: Check this case
                    print('Not sure if this can happen')

                collection.append([ 
                    label, 
                    {
                        "original": audio
                    }
                ])
            except Exception as e:
                print(e)
        gc.collect()
        return collection
