from preprocessing.audio.audio_processor import AbstractAudioPreprocessor

import librosa
import numpy as np
from tqdm import tqdm

class PitchShiftProcessor(AbstractAudioPreprocessor):
    def __init__(self, semi_tones):
        self._semi_tones = semi_tones

    def process(self, data):
        augmented_data = []
        print(f'Pitch shifting audio by {self._semi_tones} semi-tones')
        for index, file in enumerate(tqdm(data)):
            augmented_data.append([
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
            )
        return augmented_data


class PitchShiftProcessor1SemitonesUp(PitchShiftProcessor):
    def __init__(self):
        super().__init__(1)


class PitchShiftProcessor1SemitonesDown(PitchShiftProcessor):
    def __init__(self):
        super().__init__(-1)


class PitchShiftProcessor2SemitonesUp(PitchShiftProcessor):
    def __init__(self):
        super().__init__(2)


class PitchShiftProcessor2SemitonesDown(PitchShiftProcessor):
    def __init__(self):
        super().__init__(-2)
