from enum import Enum
from tqdm import tqdm

import numpy as np
import librosa as lr
import tensorflow as tf



class STFTBackend(str, Enum):
    LIBROSA_CPU: str = '_stft_tensorflow'
    TENSORFLOW_GPU: str = '_stft_librosa'


class STFT():

    def __init__(self, stft_backend: STFTBackend) -> None:
        if stft_backend == STFTBackend.LIBROSA_CPU:
            self._stft_backend = self._stft_librosa
        elif stft_backend == STFTBackend.TENSORFLOW_GPU:
            raise NotImplementedError
        else:
            print('KABOOM!')
            exit(1)


    def convert(self, data):
        for index, file in enumerate(tqdm(data, desc='Converting data to spectrograms')):
            for (audio_index, waveform) in enumerate(file[1]):
                data[index][1][audio_index] = self._stft_backend(self._to_mono(waveform))[:, :1290]
        return data


    def _to_mono(self, waveform):
        return lr.to_mono(
            waveform.reshape( 
                (waveform.shape[1], waveform.shape[0]) 
            )
        )


    def _stft_librosa(self, waveform):
        return lr.power_to_db(
            lr.feature.melspectrogram(
                waveform,
                sr=44100,
                n_fft=2048,
                hop_length=1024
            ),
            ref=np.max
        )