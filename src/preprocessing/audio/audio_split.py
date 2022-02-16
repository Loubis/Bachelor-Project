from typing import NoReturn
from .audio_processor import AbstractAudioPreprocessor

import numpy as np
from tqdm import tqdm


class AudioSlice(AbstractAudioPreprocessor):
    def __init__(self, slice_length):
        self._slice_length = slice_length


    def process(self,data):
        print(f'Sliceting audio in parts of {self._slice_length} seconds')

        new_data = []
        for file in tqdm(data):
            surplus = file[1]["original"].shape[1] % (44100 * self._slice_length)
            number_of_slices = file[1]["original"].shape[1] // (44100 * self._slice_length)

            slices = [
                np.array_split(file[1]["original"][0][surplus:], number_of_slices),
                np.array_split(file[1]["original"][1][surplus:], number_of_slices)
            ]
            for i in range(0, number_of_slices):
                new_data.append([
                    file[0], 
                    { 
                        "original": np.array([
                            slices[0][i],
                            slices[1][i],
                        ])
                    }
                ])
        return new_data


class AudioSlice_in_3_sec(AudioSlice):
    def __init__(self):
        super().__init__(3)


class AudioSlice_in_6_sec(AudioSlice):
    def __init__(self):
        super().__init__(6)


class AudioSlice_in_10_sec(AudioSlice):
    def __init__(self):
        super().__init__(10)
