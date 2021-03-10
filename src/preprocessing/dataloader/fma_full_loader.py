from .dataset_loader import AbstractDatasetLoader
import pandas as pd


class FmaFullLoader(AbstractDatasetLoader):
    
    def __init__(self):
        self._PATH = '~/Datasets/fma/'
        self._DATA_SET = 'fma_small/'
        self._META_DATA = 'fma_metadata/'


    def load(self):
        pass
