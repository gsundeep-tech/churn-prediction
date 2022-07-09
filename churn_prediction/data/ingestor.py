import logging
import pandas as pd
from churn_prediction.config.config import ModelConfig

logger = logging.getLogger(__name__)


class DataIngestor:

    def __init__(self, training_data_path, training_metadata_path) -> None:
        self.data_features_path = training_data_path
        self.data_targets_path = training_metadata_path

    def _validate_columns(self, columns, data):
        return set(columns).issubset(data.columns)

    @staticmethod
    def _get_column_dtypes():
        columns_dtypes = {}
        for col_type, columns in ModelConfig.COLUMNS.items():
            if not isinstance(columns, list):
                continue
            for column in columns:
                if col_type == 'numerical_features':
                    columns_dtypes[column] = 'float32'
                elif col_type == 'categorical_features' or col_type == 'cardinality_features':
                    columns_dtypes[column] = 'category'
        return columns_dtypes

    def _load_csv_data(self, sep, use_columns=None):
        data_features = pd.read_csv(self.data_features_path,
                                    delimiter=sep,
                                    usecols=use_columns)
        data_targets = None
        if self.data_targets_path:
            data_targets = pd.read_csv(self.data_targets_path, delimiter=sep)

        column_dtypes = self._get_column_dtypes()
        self._validate_columns(columns=column_dtypes.keys(),
                               data=data_features)

        for col, col_type in column_dtypes.items():
            try:
                data_features[col] = data_features[col].astype(col_type)
            except Exception as ex:
                logger.exception(f'Exception raise while type casting {ex}')
                raise ex
        return data_features, data_targets

    def load_data(self, type='csv', sep=',', use_columns=None):
        if type == 'csv':
            return self._load_csv_data(sep=sep, use_columns=use_columns)
        else:
            raise Exception("Only CSV with Comma Seperated is suppored")
