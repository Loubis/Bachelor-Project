from preprocessing.audio.audio_processor import AbstractAudioPreprocessor
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


class PitchShiftProcessor(AbstractAudioPreprocessor):
    def __init__(self, semi_tones):
        self._semi_tones = semi_tones

    def process(self, data):
        print(f'Pitch shifting audio by {self._semi_tones} semi-tones')
        pool = Pool(8)
        augmented_data = list(tqdm(pool.imap(self._pool_func, data), total=len(data)))
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
                    librosa.effects.pitch_shift(file[1]["original"][0], 441000, n_steps=self._semi_tones),
                    librosa.effects.pitch_shift(file[1]["original"][1], 441000, n_steps=self._semi_tones)
                ))
            }
        ]




class PitchShiftProcessor1SemitonesUp(PitchShiftProcessor):
    def __init__(self):
        super().__init__(1)


class PitchShiftProcessor1SemitonesDown(PitchShiftProcessor):
    def __init__(self):
        super().__init__(-1)
