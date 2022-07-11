from churn_prediction.model.xgboost_model import XGBoostModelPredictor, XGBoostModelTrainer


class ModelTrainerFactory:
    def __init__(self, model_type='lgbm'):
        self.model_type = model_type

    def get_trainer_object(self, save_path):
        if self.model_type == 'xgboost':
            xgboost_trainer = XGBoostModelTrainer(save_path=save_path)
            return xgboost_trainer
        else:
            raise Exception("Only xgboost model is supported")


class ModelPredictorFactory:
    def __init__(self, model_type='lgbm'):
        self.model_type = model_type

    def get_predictor_object(self, save_path):
        if self.model_type == 'xgboost':
            xgboost_predictor = XGBoostModelPredictor(save_path=save_path)
            return xgboost_predictor
        else:
            raise Exception("Only xgboost model is supported")