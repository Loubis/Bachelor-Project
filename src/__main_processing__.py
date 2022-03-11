from doctest import FAIL_FAST
from preprocessing import ModularPreprocessor, Dataset, PreprocessorModule, SeperationModel, SourceSeperationModule, STFTBackend


def main():
    pipeline = [
    ]

    preprocessor2 = ModularPreprocessor(
        dataset_path='/datashare_small/osterburg_data',
        dataset=Dataset.FMA_MEDIUM,
        preprocessor_pipeline=pipeline,
        source_seperation_module=SourceSeperationModule.SPLEETER_GPU,
        seperation_model=SeperationModel.MODEL_4_STEMS,
        keep_original=True,
        stft_backend=STFTBackend.LIBROSA_CPU,
        chunk_size=100,
        load_augmented=False
    )
    preprocessor2.run()

    preprocessor1 = ModularPreprocessor(
        dataset_path='/datashare_small/osterburg_data',
        dataset=Dataset.FMA_MEDIUM,
        preprocessor_pipeline=pipeline,
        source_seperation_module=SourceSeperationModule.OFF,
        seperation_model=SeperationModel.MODEL_2_STEMS,
        keep_original=True,
        stft_backend=STFTBackend.LIBROSA_CPU,
        chunk_size=100,
        load_augmented=False
    )
    preprocessor1.run()

    #preprocessor3 = ModularPreprocessor(
    #    dataset_path='/datashare_small/osterburg_data',
    #    dataset=Dataset.FMA_MEDIUM,
    #    preprocessor_pipeline=pipeline,
    #    source_seperation_module=SourceSeperationModule.SPLEETER_GPU,
    #    seperation_model=SeperationModel.MODEL_2_STEMS,
    #    keep_original=True,
    #    stft_backend=STFTBackend.LIBROSA_CPU,
    #    chunk_size=100,
    #    load_augmented=False
    #)
    #preprocessor3.run()


if __name__ == "__main__":
    main()
