import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer

from churn_prediction.config.config import ModelConfig
from churn_prediction.data.transform import HighCardinalityTransformer, NumericalTransformer, CategoricalTransformer
from churn_prediction.model.model import ModelTrainerFactory
from churn_prediction.utils.data_utils import DataFrameSelector


class TrainingPipeline:

    def __init__(self, data, model_save_path) -> None:
        self.data = data
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
            if not len(columns) or feature_type in ['target_feature', 'key_column']:
                continue
            if feature_type == 'cardinality_features' and ModelConfig.USE_REGION_CODE_MODEL:
                columns = ModelConfig.CARDINALITY_FEATURES_REGION_BASED_MODEL
            feature_pipeline = self._process_features(feature_type)
            pipeline_steps.append((feature_type, feature_pipeline, columns))

        # pipeline_steps.append(('target', DataFrameSelector(), [ModelConfig.COLUMNS['target_feature']]))

        if len(pipeline_steps) == 1:
            raise Exception("No Columns are defined in the config")

        return ColumnTransformer(pipeline_steps,
                                 sparse_threshold=0,
                                 verbose_feature_names_out=True)

    def __transform_data(self, pipeline, data):
        data_features, data_targets = data, data[[ModelConfig.COLUMNS.get('key_column'), ModelConfig.COLUMNS.get('target_feature')]]
        data_transformed = pipeline.fit_transform(data_features)
        column_names = pipeline.get_feature_names_out()
        data = pd.DataFrame(data_transformed, columns=column_names)
        new_column_mappings = self.__get_column_mappings()
        data.rename(columns=new_column_mappings, inplace=True)
        data = pd.merge(data, data_targets, on=ModelConfig.COLUMNS.get('key_column'))
        return data

    def __get_column_mappings(self):
        return {
            'categorical_features__vehicle_age_1-2 Year': 'categorical_features__vehicle_age_bt_1_2_Year',
            'categorical_features__vehicle_age_< 1 Year': 'categorical_features__vehicle_age_lt_1_Year',
            'categorical_features__vehicle_age_> 2 Years': 'categorical_features__vehicle_age_gt_2_Years',
            # 'target__response': ModelConfig.COLUMNS.get('target_feature')
            'id__cust_id': ModelConfig.COLUMNS.get('key_column')
        }

    def save(self, models, pipeline):
        data = {}
        data['models'] = models
        data['pipeline'] = pipeline
        data['model_type'] = ModelConfig.MODEL_TYPE
        model_path = os.path.join(self.model_save_path, 'model.pkl')
        joblib.dump(data, model_path)
    
    def generate_model_keys(self):
        model_keys = []
        generic_keys = []
        for key, value in self.data[ModelConfig.MODEL_KEY].value_counts().iteritems():
            if value > ModelConfig.MIN_DATA_POINTS:
                model_keys.append(str(key))
            else:
                generic_keys.append(str(key))
        model_keys.append('generic')
        return model_keys, generic_keys

    def train(self):
        pipeline = self._build_data_pipeline()
        model_keys, generic_keys = self.generate_model_keys()

        facotry = ModelTrainerFactory(model_type=ModelConfig.MODEL_TYPE)
        trainer = facotry.get_trainer_object(save_path=self.model_save_path)
        models = {}

        for key in model_keys:
            if key == 'generic':
                data_subset = self.data.loc[self.data[ModelConfig.MODEL_KEY].isin(generic_keys)]
            else:
                data_subset = self.data.loc[self.data[ModelConfig.MODEL_KEY] == key]
            data_subset = data_subset.loc[:, data_subset.columns != ModelConfig.MODEL_KEY]

            data_subset_transformed = self.__transform_data(pipeline=pipeline, data=data_subset)
            train_columns = set(data_subset_transformed.columns) - set([self.key_column, self.target_column])
            trained_model, f1_threshold, recall_threshold = trainer.train(data=data_subset_transformed,
                                                                          train_columns=train_columns,
                                                                          target_column=ModelConfig.COLUMNS.get('target_feature'),
                                                                          model_key=key)
            models[key] = {
                '_model': trained_model,
                'f1_threshold': f1_threshold,
                'recall_threshold': recall_threshold
            }
        self.save(models, pipeline)
