import sys
import warnings
import os
import argparse
from model.parallel_crnn import ParallelCRNN
from model.simple_cnn import SimpleCNN
from model.resnet50 import ResNet50V2

from preprocessing import ModularPreprocessor, PreprocessorModule, Dataset, SourceSeperationModule, STFTBackend


def main():
    pipeline = []
    preprocessor = ModularPreprocessor('/data/raw', Dataset.GTZAN, pipeline, SourceSeperationModule.SPLEETER_GPU, True, STFTBackend.LIBROSA_CPU, chunk_size=100)
    preprocessor.run()

    model = ParallelCRNN('/data/processed/gtzan/')
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
