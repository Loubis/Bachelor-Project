from typing import List
from enum import Enum
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
from .audio.spleeter_processor import SpleeterPreprocessor
from .audio.audio_split import AudioSplit, AudioSplit_in_3, AudioSplit_in_6, AudioSplit_in_10

from .audio.STFT import STFTBackend, STFT


def create_class_instance(classname: str):
    return globals()[classname]


class Dataset(Enum):
    GTZAN: str     = 'GtzanLoader'
    FMA_SMALL: str = 'FmaSmallLoader'
    FMA_LARGE: str = 'FmaLargeLoader'
    FMA_FULL: str  = 'FmaFullLoader'
#    BALLROM = 'ballroom'
#    MILLIONSONG = 'millionsong'



class PreprocessorModules(Enum):
    SPLEETER: str    = 'SpleeterPreprocessor'
    SPLIT_IN_3: str  = 'AudioSplit_in_3'
    SPLIT_IN_6: str  = 'AudioSplit_in_6'
    SPLIT_IN_10: str = 'AudioSplit_in_10'



class ModularPreprocessor():

    def __init__(self, dataset_path: str, dataset: Dataset, stft_backend: STFTBackend, preprocessor_pipeline: List[PreprocessorModules], chunk_size = 2) -> None:
        self._dataset: AbstractDatasetLoader = create_class_instance(dataset.value)(dataset_path)
        self._file_loader = FileLoader()
        self._preprocessor_pipeline: List[AbstractAudioPreprocessor] = []
        for processor in preprocessor_pipeline:
            self._preprocessor_pipeline.append(create_class_instance(processor.value)())
        self._stft = STFT(stft_backend)
        self._chunk_size = chunk_size


    def run(self):
        print('Loading Dataset ...')
        df = self._dataset.load()

        print('Shuffle Dataset ...')
        split = {}
        split['train'], split['validate'], split['test'] = np.split(
            df.sample(frac=1).reset_index(drop=True), 
            [int(.6*len(df)), int(.8*len(df))]
        )

        for (key, df) in split.items():
            print(f'Splitting {key} dataset into chunks ...')
            chunks = np.array_split(df, math.ceil(df.shape[0] / self._chunk_size))

            for index, chunk in enumerate(chunks):
                print('Processing Chunk ' + str(index))
                data = self._file_loader.load(chunk)

                for preprocessor in self._preprocessor_pipeline:
                    data = preprocessor.process(data)

                data = self._stft.convert(data)

                print(f'Saving {key} chunk {index}')
                audio_data = []
                labels = []
                for file in data:
                    audio = np.array(file[1])
                    audio_data.append(audio)
                    labels.append(file[0])

                np.savez(self._dataset.destination + '/arr_' + key + '_' + str(index), np.array(audio_data), np.array(labels))
                gc.collect()
            print(f'Finished {key} dataset')
        print('Finished preprocessing')
