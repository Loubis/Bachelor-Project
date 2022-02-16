from preprocessing import ModularPreprocessor, Dataset, PreprocessorModule, SeperationModel, SourceSeperationModule, STFTBackend


def main():
    pipeline = [
        PreprocessorModule.PITCH_SHIFT_2_SEMITONE_DOWN,
        PreprocessorModule.PITCH_SHIFT_2_SEMITONE_UP,
        PreprocessorModule.TIME_STRETCH_0_8,
        PreprocessorModule.TIME_STRETCH_1_2,
    ]

    preprocessor = ModularPreprocessor(
        dataset_path='/data',
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
