from typing import List
from enum import Enum
from pandas.io.parsers import PythonParser
from tqdm import tqdm

import numpy as np
import pandas as pd
import math
import gc
import json

from .dataloader.dataset_loader import AbstractDatasetLoader
from .dataloader.file_loader import FileLoader
from .dataloader.gtzan_loader import GtzanLoader
from .dataloader.fma_small_loader import FmaSmallLoader
from .dataloader.fma_medium_loader import FmaMediumLoader

from .audio.audio_processor import AbstractAudioPreprocessor

from .audio.spleeter_processor import SpleeterCPUPreprocessor, SpleeterGPUPreprocessor

from .audio.audio_split import AudioSplit_in_3, AudioSplit_in_6, AudioSplit_in_10

from .audio.STFT import LibrosaCPUSTFT


def create_class_instance(classname: str):
    return globals()[classname]


class STFTBackend(Enum):
    LIBROSA_CPU: str = LibrosaCPUSTFT.__name__


class Dataset(Enum):
    GTZAN: str     = GtzanLoader.__name__
    FMA_SMALL: str = FmaSmallLoader.__name__
    FMA_MEDIUM: str = FmaMediumLoader.__name__


class PreprocessorModule(Enum):
    SPLIT_IN_3: str  = AudioSplit_in_3.__name__
    SPLIT_IN_6: str  = AudioSplit_in_6.__name__
    SPLIT_IN_10: str = AudioSplit_in_10.__name__


class SourceSeperationModule(Enum):
    SPLEETER_CPU: str = SpleeterCPUPreprocessor.__name__
    SPLEETER_GPU: str = SpleeterGPUPreprocessor.__name__
    OFF: bool = False


class SeperationModel(Enum):
    MODEL_2_STEMS = 'spleeter:2stems'
    MODEL_4_STEMS = 'spleeter:4stems'
    MODEL_5_STEMS = 'spleeter:5stems'


class ModularPreprocessor():

    def __init__(
        self, 
        dataset_path: str, dataset: Dataset, 
        preprocessor_pipeline: List[PreprocessorModule], 
        source_seperation_module: SourceSeperationModule,
        seperation_model=SeperationModel.MODEL_2_STEMS,
        keep_original = True,
        stft_backend = STFTBackend,
        chunk_size = 100
    ):
        pipeline_str = ''

        self._chunk_size = chunk_size
        self._file_loader = FileLoader()

        self._preprocessor_pipeline: List[AbstractAudioPreprocessor] = []
        for processor in preprocessor_pipeline:
            self._preprocessor_pipeline.append(create_class_instance(processor.value)())
            pipeline_str += f'_{processor.value}'
        
        if source_seperation_module is not SourceSeperationModule.OFF:
            self._source_seperation_module = create_class_instance(source_seperation_module.value)(keep_original, seperation_model.value)
            pipeline_str += f'_{source_seperation_module.value}_{seperation_model.value}'
            if keep_original:
                pipeline_str += f'_keepOriginal'
        else:
            self._source_seperation_module = False
            pipeline_str += f'_noSeperation'

        self._stft = create_class_instance(stft_backend.value)()
        pipeline_str += f'_{stft_backend.value}'

        self._dataset: AbstractDatasetLoader = create_class_instance(dataset.value)(dataset_path, pipeline_str)


    def run(self):
        print('Loading Dataset ...')
        df, metadata = self._dataset.load()

        print('Shuffle Dataset ...')
        split = {}
        split['train'], split['validate'], split['test'] = np.split(
            df.sample(frac=1).reset_index(drop=True), 
            [int(0.6*len(df)), int(0.8*len(df))]
        )

        metadata_created = False
        for (key, df) in split.items():
            print(f'Splitting {key} dataset into chunks ...')
            chunks = np.array_split(df, math.ceil(df.shape[0] / self._chunk_size))
 
            for index, chunk in enumerate(chunks):
                print(f'Processing Chunk {str(index)}')
                print(f'Loading Files')
                data = self._file_loader.load(chunk)

                print('Processing pipeline')
                for preprocessor in self._preprocessor_pipeline:
                    data = preprocessor.process(data)

                if self._source_seperation_module:
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

                    if not metadata_created:
                        metadata_created = True
                        metadata['data_shape'] = (audio.shape[1],audio.shape[2])
                        metadata['split_count'] = audio.shape[0] 
                        with open(f'{self._dataset.destination}/metadata.json', 'w') as outfile:
                            json.dump(metadata, outfile)

                np.savez(f'{self._dataset.destination}/arr_{key}_{str(index)}', np.array(audio_data), np.array(labels))
                gc.collect()

            print(f'Finished {key} dataset')
        print('Finished preprocessing')
