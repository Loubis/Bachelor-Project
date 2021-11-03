from enum import Enum
from tqdm import tqdm

from spleeter.separator import Separator
from spleeter.audio import STFTBackend

from .audio_processor import AbstractAudioPreprocessor


class SpleeterPreprocessor(AbstractAudioPreprocessor):

    def __init__(self, stft_backend: STFTBackend, keep_original, seperation_model):
        self._keep_original = keep_original
        self._seperator = Separator(seperation_model, stft_backend, multiprocess=True)


    def process(self, data):
        for index, file in enumerate(tqdm(data)):
            waveform = file[1][0]
            try:
                prediction = self._seperator.separate(waveform, "")
                if self._keep_original:
                    data[index][1].extend(prediction.values())
                else:
                    data[index][1] = list(prediction.values())
            except KeyboardInterrupt:
                exit(1)
            except Exception as e:
                print(e)
        return data



class SpleeterGPUPreprocessor(SpleeterPreprocessor):
    def __init__(self, keep_original, seperation_model):
        super().__init__(STFTBackend.TENSORFLOW, keep_original, seperation_model)


class SpleeterCPUPreprocessor(SpleeterPreprocessor):
    def __init__(self, keep_original, seperation_model):
        super().__init__(STFTBackend.LIBROSA, keep_original, seperation_model)


