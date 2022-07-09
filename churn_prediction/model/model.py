from salary_prediction.model.lgbm_model import LGBMModelPredictor, LGBMModelTrainer
from salary_prediction.model.catboost_model import CatBoostModelTrainer, CatBoostModelPredictory


class ModelTrainerFactory:

    def __init__(self, model_type='lgbm'):
        self.model_type = model_type

    def get_trainer_object(self, raw_data, transformed_data, train_columns,
                           target_column, save_path):
        if self.model_type == 'lgbm':
            lgbm_trainer = LGBMModelTrainer(raw_data=raw_data,
                                            transformed_data=transformed_data,
                                            train_columns=train_columns,
                                            target_column=target_column,
                                            save_path=save_path)
            return lgbm_trainer
        elif self.model_type == 'catboost':
            catboost_trainer = CatBoostModelTrainer(
                raw_data=raw_data,
                transformed_data=transformed_data,
                train_columns=train_columns,
                target_column=target_column,
                save_path=save_path)
            return catboost_trainer
        else:
            raise Exception("Only lgbm, catboost models is supported")


class ModelPredictorFactory:

    def __init__(self, model_type='lgbm'):
        self.model_type = model_type

    def get_predictor_object(self, raw_data, transformed_data, predict_columns,
                             model, save_path):
        if self.model_type == 'lgbm':
            lgbm_predictor = LGBMModelPredictor(
                raw_data=raw_data,
                transformed_data=transformed_data,
                prediction_columns=predict_columns,
                model=model,
                save_path=save_path)
            return lgbm_predictor
        elif self.model_type == 'catboost':
            catboost_predictor = CatBoostModelPredictory(
                raw_data=raw_data,
                transformed_data=transformed_data,
                prediction_columns=predict_columns,
                model=model,
                save_path=save_path)
            return catboost_predictor
        else:
            raise Exception("Only lgbm, catboost model is supported")