from enum import Enum
from tqdm import tqdm

from spleeter.separator import Separator
from spleeter.audio import STFTBackend

from .audio_processor import AbstractAudioPreprocessor


class SeperationModel(Enum):
    MODEL_2_STEMS = 'spleeter:2stems'
    MODEL_4_STEMS = 'spleeter:4stems'
    MODEL_5_STEMS = 'spleeter:5stems'
    MODEL_2_STEMS_16_KHZ = 'spleeter:2stems-16kHz'
    MODEL_4_STEMS_16_KHZ = 'spleeter:4stems-16kHz'
    MODEL_5_STEMS_16_KHZ = 'spleeter:5stems-16kHz'


class SpleeterPreprocessor(AbstractAudioPreprocessor):

    def __init__(self, seperation_model=SeperationModel.MODEL_2_STEMS.value):
        self._seperator = Separator(seperation_model, STFTBackend.TENSORFLOW, multiprocess=False)

    def process(self, data):
        for index, file in enumerate(tqdm(data, desc='Splitting Files')):
            waveform = file[1][0]
            try:
                prediction = self._seperator.separate(waveform, "")
                data[index][1].extend(prediction.values())
            except KeyboardInterrupt:
                exit(1)
            except Exception as e:
                print(e)
        return data
