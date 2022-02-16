from abc import ABC, abstractmethod

class AbstractModel(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def train(self):
        pass


    @abstractmethod
    def evaluate(self):
        pass
