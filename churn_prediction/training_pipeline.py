import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer

from salary_prediction.config.config import ModelConfig
from salary_prediction.data.transform import HighCardinalityTransformer, NumericalTransformer, CategoricalTransformer
from salary_prediction.model.model import ModelTrainerFactory
from salary_prediction.utils.data_utils import DataFrameSelector


class TrainingPipeline:

    def __init__(self, data_features, data_targets, model_save_path) -> None:
        self.data_features = data_features
        self.data_targets = data_targets
        self.model_save_path = model_save_path
        self.key_column = ModelConfig.COLUMNS["key_column"]
        self.target_column = ModelConfig.COLUMNS["target_feature"]

    def _process_features(self, feature_type):
        if feature_type == 'numerical_features':
            return NumericalTransformer.transform()
        elif feature_type == 'categorical_features':
            return CategoricalTransformer.transform()
        elif feature_type == 'cardinality_features':
            return HighCardinalityTransformer.transform()

    def _build_data_pipeline(self):
        pipeline_steps = [('id', DataFrameSelector(),
                           [ModelConfig.COLUMNS['key_column']])]
        for feature_type, columns in ModelConfig.COLUMNS.items():
            if not len(columns) or feature_type in [
                    'target_feature', 'key_column'
            ]:
                continue
            feature_pipeline = self._process_features(feature_type)
            pipeline_steps.append((feature_type, feature_pipeline, columns))

        if len(pipeline_steps) == 1:
            raise Exception("No Columns are defined in the config")

        return ColumnTransformer(pipeline_steps,
                                 sparse_threshold=0,
                                 verbose_feature_names_out=True)

    def __transform_data(self, pipeline):
        data_transformed = pipeline.fit_transform(self.data_features)
        column_names = pipeline.get_feature_names_out()
        data = pd.DataFrame(data_transformed, columns=column_names)
        new_key_column = self.__get_key_column_name()
        data.rename(columns={new_key_column: self.key_column}, inplace=True)
        data = pd.merge(data, self.data_targets, on=self.key_column)
        return data

    def __get_key_column_name(self):
        return f'id__{self.key_column}'

    def save(self, model, pipeline):
        data = {}
        data['model'] = model
        data['pipeline'] = pipeline
        data['model_type'] = ModelConfig.MODEL_TYPE
        model_path = os.path.join(self.model_save_path, 'model.pkl')
        joblib.dump(data, model_path)

    def train(self):
        pipeline = self._build_data_pipeline()
        data = self.__transform_data(pipeline=pipeline)

        train_columns = set(data.columns) - set(
            [self.key_column, self.target_column])
        raw_data = pd.merge(self.data_features,
                            self.data_targets,
                            on=self.key_column)
        facotry = ModelTrainerFactory(model_type=ModelConfig.MODEL_TYPE)
        trainer = facotry.get_trainer_object(raw_data=raw_data,
                                             transformed_data=data,
                                             train_columns=train_columns,
                                             target_column=self.target_column,
                                             save_path=self.model_save_path)
        model = trainer.train()
        self.save(model, pipeline)
