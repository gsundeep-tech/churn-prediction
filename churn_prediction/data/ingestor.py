import logging
import pandas as pd
from churn_prediction.config.config import ModelConfig

logger = logging.getLogger(__name__)


class DataIngestor:
    def __init__(self, data_path, metadata_path=None) -> None:
        self.data_path = data_path
        self.metadata_path = metadata_path

    def _validate_columns(self, columns, data):
        return set(columns).issubset(data.columns)

    def _get_column_dtypes(self):
        available_columns = None
        if self.metadata_path:
            metadata = pd.read_excel(self.metadata_path)
            metadata = metadata.loc[metadata['Required'] == 'Yes']
            available_columns = list(metadata['Field Name'])
        columns_dtypes = {}
        for col_type, columns in ModelConfig.COLUMNS.items():
            if not isinstance(columns, list):
                continue
            for column in columns:
                if available_columns and column not in available_columns:
                    raise Exception("Invalid Column in the configuration")
                if col_type == 'numerical_features':
                    columns_dtypes[column] = 'float32'
                elif col_type == 'categorical_features' or col_type == 'cardinality_features':
                    columns_dtypes[column] = 'category'
        return columns_dtypes

    def _load_csv_data(self, sep, use_columns=None):
        data = pd.read_csv(self.data_path,
                           delimiter=sep,
                           usecols=use_columns)

        column_dtypes = self._get_column_dtypes()
        self._validate_columns(columns=column_dtypes.keys(), data=data)

        for col, col_type in column_dtypes.items():
            try:
                data[col] = data[col].apply(str)
                data[col] = data[col].astype(col_type)
            except Exception as ex:
                logger.exception(f'Exception raise while type casting {ex}')
                raise ex
        return data

    def load_data(self, type='csv', sep=',', use_columns=None):
        if type == 'csv':
            return self._load_csv_data(sep=sep, use_columns=use_columns)
        else:
            raise Exception("Only CSV with Comma Seperated is suppored")
