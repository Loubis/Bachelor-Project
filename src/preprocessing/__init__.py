from typing import List
from enum import Enum
from multiprocessing import Process


import numpy as np
import pandas as pd
import math
import gc
import json

from preprocessing.audio.time_stretch_processor import TimeStretchProcessorFactor0_8, TimeStretchProcessorFactor1_2

from .dataloader.dataset_loader import AbstractDatasetLoader
from .dataloader.file_loader import FileLoader
from .dataloader.gtzan_loader import GtzanLoader
from .dataloader.fma_small_loader import FmaSmallLoader
from .dataloader.fma_medium_loader import FmaMediumLoader
from .audio.audio_processor import AbstractAudioPreprocessor
from .audio.spleeter_processor import SpleeterCPUPreprocessor, SpleeterGPUPreprocessor
from .audio.audio_split import AudioSlice_in_3_sec, AudioSlice_in_6_sec, AudioSlice_in_10_sec
from preprocessing.audio.pitch_shift_processor import PitchShiftProcessor1SemitonesDown, PitchShiftProcessor1SemitonesUp, PitchShiftProcessor2SemitonesDown, PitchShiftProcessor2SemitonesUp
from .audio.STFT import LibrosaCPUSTFT


class STFTBackend(Enum):
    LIBROSA_CPU: str = LibrosaCPUSTFT.__name__


class Dataset(Enum):
    GTZAN: str     = GtzanLoader.__name__
    FMA_SMALL: str = FmaSmallLoader.__name__
    FMA_MEDIUM: str = FmaMediumLoader.__name__


class PreprocessorModule(Enum):
    AUDIO_SLICE_IN_3_SEC:        str = AudioSlice_in_3_sec.__name__
    AUDIO_SLICE_IN_6_SEC:        str = AudioSlice_in_6_sec.__name__
    AUDIO_SLICE_IN_10_SEC:       str = AudioSlice_in_10_sec.__name__
    PITCH_SHIFT_1_SEMITONE_UP:   str = PitchShiftProcessor1SemitonesUp.__name__
    PITCH_SHIFT_2_SEMITONE_UP:   str = PitchShiftProcessor2SemitonesUp.__name__
    PITCH_SHIFT_1_SEMITONE_DOWN: str = PitchShiftProcessor1SemitonesDown.__name__
    PITCH_SHIFT_2_SEMITONE_DOWN: str = PitchShiftProcessor2SemitonesDown.__name__
    TIME_STRETCH_1_2:            str = TimeStretchProcessorFactor1_2.__name__
    TIME_STRETCH_0_8:            str = TimeStretchProcessorFactor0_8.__name__


class SourceSeperationModule(Enum):
    SPLEETER_CPU: str = SpleeterCPUPreprocessor.__name__
    SPLEETER_GPU: str = SpleeterGPUPreprocessor.__name__
    OFF: bool = False


class SeperationModel(Enum):
    MODEL_2_STEMS = 'spleeter:2stems'
    MODEL_4_STEMS = 'spleeter:4stems'
    MODEL_5_STEMS = 'spleeter:5stems'


# Helper to create Class Instances through name as string
def create_class_instance(classname: str):
    return globals()[classname]

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
        split, metadata = self._dataset.load()

        metadata_created = False
        for (key, df) in split.items():
            print(f'Splitting {key} dataset into chunks ...')
            chunks = np.array_split(df, math.ceil(df.shape[0] / self._chunk_size))

            for index, chunk in enumerate(chunks):
                print(f'Processing Chunk {str(index)}')
                print(f'Loading Files')
                data = self._file_loader.load(chunk)
                augmented_data = []

                processes = []
                print('Processing pipeline')
                for preprocessor in self._preprocessor_pipeline:
                    augmented_data = augmented_data + preprocessor.process(data)

                print('Merge data and augmented data')
                data = data + augmented_data
                splitter = create_class_instance(PreprocessorModule.AUDIO_SLICE_IN_10_SEC.value)()
                data = splitter.process(data)

                if self._source_seperation_module:
                    print('Processing source seperation')
                    data = self._source_seperation_module.process(data)

                print('Converting to spectrogram')
                data = self._stft.convert(data)

                print(f'Saving {key} chunk {index}')
                audio_data = []
                labels = []
                for file in data:
                    audio = file[1]
                    audio_data.append(audio)
                    labels.append(file[0])

                    if not metadata_created:
                        metadata_created = True
                        metadata['data_shape'] = (list(audio.values())[0].shape[0], list(audio.values())[0].shape[1])
                        metadata['split_count'] = len(list(audio.values()))
                        with open(f'{self._dataset.destination}/metadata.json', 'w') as outfile:
                            json.dump(metadata, outfile)

                np.savez(f'{self._dataset.destination}/arr_{key}_{str(index)}', np.array(audio_data), np.array(labels))
                gc.collect()

            print(f'Finished {key} dataset')
        print('Finished preprocessing')
