from typing import List
from enum import Enum
from pandas.io.parsers import PythonParser
from tqdm import tqdm

import numpy as np
import pandas as pd
import math
import gc

# TODO: fixing import
from .dataloader.dataset_loader import AbstractDatasetLoader
from .dataloader.file_loader import FileLoader
from .dataloader.gtzan_loader import GtzanLoader
from .dataloader.fma_small_loader import FmaSmallLoader
from .dataloader.fma_large_loader import FmaLargeLoader
from .dataloader.fma_full_loader import FmaFullLoader

from .audio.audio_processor import AbstractAudioPreprocessor

from .audio.spleeter_processor import SpleeterCPUPreprocessor
from .audio.spleeter_processor import SpleeterGPUPreprocessor

from .audio.audio_split import AudioSplit_in_3, AudioSplit_in_6, AudioSplit_in_10

from .audio.STFT import LibrosaCPUSTFT


def create_class_instance(classname: str):
    return globals()[classname]


class STFTBackend(Enum):
    LIBROSA_CPU: str = 'LibrosaCPUSTFT'
    TENSORFLOW_GPU: str = 'TensorflowGPUSTFT'


class Dataset(Enum):
    GTZAN: str     = 'GtzanLoader'
    FMA_SMALL: str = 'FmaSmallLoader'
    FMA_LARGE: str = 'FmaLargeLoader'
    FMA_FULL: str  = 'FmaFullLoader'


class PreprocessorModule(Enum):
    SPLIT_IN_3: str  = 'AudioSplit_in_3'
    SPLIT_IN_6: str  = 'AudioSplit_in_6'
    SPLIT_IN_10: str = 'AudioSplit_in_10'


class SourceSeperationModule(Enum):
    SPLEETER_CPU: str = 'SpleeterCPUPreprocessor'
    SPLEETER_GPU: str = 'SpleeterGPUPreprocessor'
    NUSSL: str = 'NusslPreprocessor'
    OFF: bool = False


class ModularPreprocessor():

    def __init__(
        self, 
        dataset_path: str, dataset: Dataset, 
        preprocessor_pipeline: List[PreprocessorModule], 
        source_seperation_module: SourceSeperationModule, keep_origional = True,
        stft_backend = STFTBackend,
        chunk_size = 100
    ):
        self._dataset: AbstractDatasetLoader = create_class_instance(dataset.value)(dataset_path)
        self._chunk_size = chunk_size
        self._file_loader = FileLoader()

        self._preprocessor_pipeline: List[AbstractAudioPreprocessor] = []
        for processor in preprocessor_pipeline:
            self._preprocessor_pipeline.append(create_class_instance(processor.value)())

        self._source_seperation_module = create_class_instance(source_seperation_module.value)(keep_origional)
        self._stft = create_class_instance(stft_backend.value)()


    def run(self):
        print('Loading Dataset ...')
        df = self._dataset.load()

        print('Shuffle Dataset ...')
        split = {}
        split['train'], split['validate'], split['test'] = np.split(
            df.sample(frac=1).reset_index(drop=True), 
            [int(0.6*len(df)), int(0.8*len(df))]
        )

        for (key, df) in split.items():
            print(f'Splitting {key} dataset into chunks ...')
            chunks = np.array_split(df, math.ceil(df.shape[0] / self._chunk_size))
 
            for index, chunk in enumerate(chunks):
                print(f'Processing Chunk {str(index)}')
                data = self._file_loader.load(chunk)

                print('Processing pipeline')
                for preprocessor in self._preprocessor_pipeline:
                    data = preprocessor.process(data)

                print('Processing source seperation')
                data = self._source_seperation_module.process(data)

                print('Converting to spectrogram')
                data = self._stft.convert(data)

                print(f'Saving {key} chunk {index}')
                audio_data = []
                labels = []
                for file in data:
                    audio = np.array(file[1])
                    audio_data.append(audio)
                    labels.append(file[0])
                np.savez(f'{self._dataset.destination}/arr_{key}_{str(index)}', np.array(audio_data), np.array(labels))
                gc.collect()

            print(f'Finished {key} dataset')
        print('Finished preprocessing')
