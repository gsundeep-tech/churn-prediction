from abc import ABC, abstractmethod


class AbstractTrainerModel(ABC):

    def __init__(self, raw_data, transformed_data, train_columns,
                 target_column, save_path):
        self.raw_data = raw_data
        self.transformed_data = transformed_data
        self.train_columns = train_columns
        self.target_column = target_column
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

    def __init__(self, raw_data, transformed_data, prediction_columns, model,
                 save_path):
        self.raw_data = raw_data
        self.transformed_data = transformed_data
        self.prediction_columns = prediction_columns
        self.model = model
        self.save_path = save_path

    @abstractmethod
    def predict(self):
        pass