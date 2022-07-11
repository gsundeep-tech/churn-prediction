import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from churn_prediction.config.config import ModelConfig


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, *args, **kwargs):
        return args[0]


def catboost_data_preprocess(raw_data, transformed_data):
    data = pd.merge(raw_data,
                    transformed_data[[
                        'jobId', 'numerical_features__yearsExperience',
                        'numerical_features__milesFromMetropolis'
                    ]],
                    on=ModelConfig.COLUMNS['key_column'])
    data.drop(['yearsExperience', 'milesFromMetropolis'], axis=1, inplace=True)
    data.rename(columns={
        'numerical_features__yearsExperience':
        'yearsExperience',
        'numerical_features__milesFromMetropolis':
        'milesFromMetropolis'
    },
                inplace=True)
    return data