from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import numpy as np
import os

class AbstractDatasetLoader(ABC):
    def __init__(self):
        self._pd = pd
        self._np = np
        self._os = os
        self.destination = ''

    @abstractmethod
    def load(self):
        ...