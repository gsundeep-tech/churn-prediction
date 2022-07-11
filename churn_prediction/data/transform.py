from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


class AbstractDataTransformer(ABC):

    def __init__(self, data, columns):
        self.data = data
        self.columns = columns


class NumericalTransformer(AbstractDataTransformer):

    @staticmethod
    def transform():
        return Pipeline([('num_constant_imputer',
                          SimpleImputer(strategy='constant', fill_value=0)),
                         ('robust_scaler', RobustScaler())])


class CategoricalTransformer(AbstractDataTransformer):

    @staticmethod
    def transform():
        return Pipeline([
            ("cat_constant_imputer",
             SimpleImputer(missing_values='NONE',
                           strategy="constant",
                           fill_value='OTHER')),
            ("encoder", OneHotEncoder()),
        ])


class HighCardinalityTransformer(AbstractDataTransformer):

    @staticmethod
    def transform():
        return Pipeline([
            ("cardinality_constant_imputer",
             SimpleImputer(missing_values='NONE',
                           strategy="constant",
                           fill_value='OTHER')),
            ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])