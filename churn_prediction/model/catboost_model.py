import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from salary_prediction.config.config import ModelConfig

from salary_prediction.model.abstract_model import AbstractTrainerModel, AbstractPredictorModel
from salary_prediction.utils.data_utils import catboost_data_preprocess


class CatBoostModelTrainer(AbstractTrainerModel):

    def __init__(self, raw_data, transformed_data, train_columns,
                 target_column, save_path):
        super().__init__(raw_data, transformed_data, train_columns,
                         target_column, save_path)
        self.train_columns = ModelConfig.TRAIN_FEATURES

    def train(self):
        data = catboost_data_preprocess(self.raw_data, self.transformed_data)
        X_train, X_valid, y_train, y_valid = train_test_split(
            data[self.train_columns],
            data[self.target_column],
            test_size=0.20,
            random_state=31)
        model = self._train_model(X_train, X_valid, y_train, y_valid)
        self.explain(model)
        self.plot(model)
        return model

    def _train_model(self, X_train, X_valid, y_train, y_valid):
        categorical_features = ModelConfig.COLUMNS.get(
            'categorical_features', []) + ModelConfig.COLUMNS.get(
                'cardinality_features', [])
        catboost_model = CatBoostRegressor(loss_function='RMSE',
                                           random_seed=31,
                                           iterations=200,
                                           cat_features=categorical_features)
        catboost_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

        return catboost_model

    def explain(self, model):
        feature_importance = model.get_feature_importance()
        feature_importance_img_path = os.path.join(self.save_path,
                                                   'feature_importance.png')
        plt.bar(range(len(self.train_columns)),
                feature_importance,
                tick_label=self.train_columns)

        plt.savefig(feature_importance_img_path)

    def plot(self, model):
        eval_results_path = os.path.join(self.save_path, 'eval_results.json')
        eval_results = model.get_evals_result()
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f)


class CatBoostModelPredictory(AbstractPredictorModel):

    def __init__(self, raw_data, transformed_data, prediction_columns, model,
                 save_path):
        super().__init__(raw_data, transformed_data, prediction_columns, model,
                         save_path)
        self.prediction_columns = ModelConfig.TRAIN_FEATURES

    def predict(self):
        data = catboost_data_preprocess(raw_data=self.raw_data,
                                        transformed_data=self.transformed_data)
        predictions = self.model.predict(data[self.prediction_columns])
        data['predictions'] = predictions
        return data