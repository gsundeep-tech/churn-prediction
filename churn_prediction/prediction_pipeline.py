import os
import logging
import joblib

import pandas as pd
from churn_prediction.config.config import ModelConfig
from churn_prediction.model.model import ModelPredictorFactory

logger = logging.getLogger(__name__)


class PredictionPipeline:

    def __init__(self, data, model_path, save_path):
        self.data = data
        self.model_path = model_path
        self.save_path = save_path
        self.key_column = ModelConfig.COLUMNS["key_column"]

    def _load_model(self):
        try:
            data = joblib.load(self.model_path)
            return data['models'], data['pipeline'], data['model_type']
        except Exception as ex:
            logger.exception("Exception while loading the model")

    def __get_column_mappings(self):
        return {
            'categorical_features__vehicle_age_1-2 Year': 'categorical_features__vehicle_age_bt_1_2_Year',
            'categorical_features__vehicle_age_< 1 Year': 'categorical_features__vehicle_age_lt_1_Year',
            'categorical_features__vehicle_age_> 2 Years': 'categorical_features__vehicle_age_gt_2_Years',
            # 'target__response': ModelConfig.COLUMNS.get('target_feature')
            'id__cust_id': ModelConfig.COLUMNS.get('key_column')
        }

    def _save_predictions(self, data):
        save_file = os.path.join(self.save_path, "predictions.csv")
        data[[ModelConfig.COLUMNS.get('key_column'), 'predictions']].to_csv(save_file, index=False)
    
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

    def predict(self):
        model, pipeline, model_type = self._load_model()
        model_keys, generic_keys = self.generate_model_keys()

        factory = ModelPredictorFactory(model_type=model_type)
        predictor = factory.get_predictor_object(save_path=self.save_path)

        results = []

        for key in model_keys:
            if key == 'generic':
                data_subset = self.data.loc[self.data[ModelConfig.MODEL_KEY].isin(generic_keys)]
            else:
                data_subset = self.data.loc[self.data[ModelConfig.MODEL_KEY] == key]
            data_subset = data_subset.loc[:, data_subset.columns != ModelConfig.MODEL_KEY]

            region_model_data = model.get(key, 'generic')
            region_model = region_model_data['_model']
            f1_threshold = region_model_data['f1_threshold']
            recall_threshold = region_model_data['recall_threshold']

            data_transformed = pipeline.transform(data_subset)
            column_names = pipeline.get_feature_names_out()

            data = pd.DataFrame(data_transformed, columns=column_names)
            column_names_mapping = self.__get_column_mappings()
            data.rename(columns=column_names_mapping, inplace=True)

            prediction_cols = set(data.columns) - set([self.key_column])
        
            data = predictor.predict(model=region_model, data=data, prediction_columns=prediction_cols,
                                     f1_threshold=f1_threshold, recall_threshold=recall_threshold)

            results.append(data)

        data = pd.concat(results)

        self._save_predictions(data)