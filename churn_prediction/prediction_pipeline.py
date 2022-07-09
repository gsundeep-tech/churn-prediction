import os
import logging
import joblib

import pandas as pd
from salary_prediction.config.config import ModelConfig
from salary_prediction.model.model import ModelPredictorFactory

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
            return data['model'], data['pipeline'], data['model_type']
        except Exception as ex:
            logger.exception("Exception while loading the model")

    def __get_key_column_name(self):
        return f'id__{self.key_column}'

    def _save_predictions(self, data):
        save_file = os.path.join(self.save_path, "predictions.csv")
        data[['jobId', 'predictions']].to_csv(save_file, index=False)

    def predict(self):
        model, pipeline, model_type = self._load_model()
        data_transformed = pipeline.transform(self.data)
        column_names = pipeline.get_feature_names_out()

        data = pd.DataFrame(data_transformed, columns=column_names)
        new_key_column = self.__get_key_column_name()
        data.rename(columns={new_key_column: self.key_column}, inplace=True)

        prediction_cols = set(data.columns) - set([self.key_column])
        factory = ModelPredictorFactory(model_type=model_type)
        predictor = factory.get_predictor_object(
            raw_data=self.data,
            transformed_data=data,
            predict_columns=prediction_cols,
            model=model,
            save_path=self.save_path)
        data = predictor.predict()

        self._save_predictions(data)