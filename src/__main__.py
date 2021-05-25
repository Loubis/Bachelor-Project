import sys
import warnings
import os
import argparse
from model.parallel_crnn import ParallelCRNN
from model.simple_cnn import SimpleCNN
from model.resnet50 import ResNet50V2

from preprocessing import ModularPreprocessor, PreprocessorModules, Dataset
from preprocessing.audio.STFT import STFTBackend

def main():
    #pipeline = [PreprocessorModules.SPLEETER]
    #preprocessor = ModularPreprocessor('/data', Dataset.GTZAN, STFTBackend.LIBROSA_CPU, pipeline, chunk_size=100)
    #preprocessor.run()

    model = ParallelCRNN('/data/processed/gtzan/')
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
 
