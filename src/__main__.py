import sys
import warnings
import os
import argparse
from model.parallel_crnn import ParallelCRNN
from model.simple_cnn import SimpleCNN

from preprocessing import ModularPreprocessor, PreprocessorModules, Dataset
from preprocessing.audio.STFT import STFTBackend

def main():
    #pipeline = [PreprocessorModules.SPLIT_IN_10 ,PreprocessorModules.SPLEETER]
    #preprocessor = ModularPreprocessor('/data', Dataset.GTZAN, STFTBackend.LIBROSA_CPU, pipeline, chunk_size=100)
    #preprocessor.run()

    model = SimpleCNN()
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
 
