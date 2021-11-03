from model.parallel_crnn import ParallelCRNN

from preprocessing import ModularPreprocessor, Dataset, PreprocessorModule, SeperationModel, SourceSeperationModule, STFTBackend


def main():
    pipeline = [
        PreprocessorModule.SPLIT_IN_3
    ]

    preprocessor = ModularPreprocessor(
        dataset_path='/datashare_small/osterburg_data',
        dataset=Dataset.FMA_MEDIUM,
        preprocessor_pipeline=pipeline,
        source_seperation_module=SourceSeperationModule.SPLEETER_GPU,
        seperation_model=SeperationModel.MODEL_4_STEMS,
        keep_original=True,
        stft_backend=STFTBackend.LIBROSA_CPU,
        chunk_size=100
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
