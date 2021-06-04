from typing import NoReturn
from .audio_processor import AbstractAudioPreprocessor

import numpy as np



class AudioSplit(AbstractAudioPreprocessor):

    def __init__(self, number_of_splits):
        self._number_of_splits = number_of_splits


    def process(self,data):
        new_data = list()
        for file in data:
            new_data.extend([ [file[0], [split]] for split in np.split(file[1][0], self._number_of_splits)])
        return new_data


class AudioSplit_in_3(AudioSplit):

    def __init__(self):
        super().__init__(3)


class AudioSplit_in_6(AudioSplit):

    def __init__(self):
        super().__init__(6)


class AudioSplit_in_10(AudioSplit):
    
    def __init__(self):
        super().__init__(10)
