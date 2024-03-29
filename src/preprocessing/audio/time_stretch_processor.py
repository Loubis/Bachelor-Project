from preprocessing.audio.audio_processor import AbstractAudioPreprocessor
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

class TimeStretchProcessor(AbstractAudioPreprocessor):
    def __init__(self, stretch_rate):
        self._stretch_rate = stretch_rate


    def process(self, data):
        print(f'Time stretching audio by rate of {self._stretch_rate}')
        pool = Pool(8)
        augmented_data =  list(tqdm(pool.imap(self._pool_func, data), total=len(data)))
        pool.close()
        pool.join()
        return augmented_data       


    def _pool_func(self, file):
        return [
                # Label
                file[0],
                # Audio
                # file[1]["original"][0]
                #      |      |      |
                #      |      |      |__ Stereo Channel
                #      |      |__ Sepearation Channel
                #      |__ Audio Dictionary
                {
                    "original": np.stack((
                        librosa.effects.time_stretch(file[1]["original"][0], rate=self._stretch_rate),
                        librosa.effects.time_stretch(file[1]["original"][1], rate=self._stretch_rate)
                    ))
                }
        ]


class TimeStretchProcessorFactor1_1(TimeStretchProcessor):
    def __init__(self):
        super().__init__(1.1)


class TimeStretchProcessorFactor0_9(TimeStretchProcessor):
    def __init__(self):
        super().__init__(0.9)
