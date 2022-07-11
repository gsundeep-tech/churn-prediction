from abc import ABC, abstractmethod


class AbstractTrainerModel(ABC):

    def __init__(self, save_path):
        self.save_path = save_path

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def explain(self):
        ...

    @abstractmethod
    def plot(self):
        ...


class AbstractPredictorModel(ABC):

    def __init__(self, save_path):
        self.save_path = save_path

    @abstractmethod
    def predict(self):
        pass