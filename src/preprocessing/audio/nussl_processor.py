from enum import Enum
from tqdm import tqdm

from spleeter.separator import Separator
from spleeter.audio import STFTBackend

from .audio_processor import AbstractAudioPreprocessor


class SeperationModel(Enum):
    ...

class NusslPreprocessor(AbstractAudioPreprocessor):

    def __init__(self):
        ...


    def process(self, data):
        ...
