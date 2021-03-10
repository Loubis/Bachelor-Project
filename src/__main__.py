import sys
import warnings
import os
import argparse
from model.parallel_crnn import ParallelCRNN

from preprocessing import ModularPreprocessor, PreprocessorModules, Dataset
from preprocessing.audio.STFT import STFTBackend

def main():
    #pipeline = [PreprocessorModules.SPLEETER]
    #preprocessor = ModularPreprocessor('/data', Dataset.FMA_SMALL, STFTBackend.LIBROSA_CPU, pipeline, chunk_size=100)
    #preprocessor.run()

    model = ParallelCRNN()
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
 