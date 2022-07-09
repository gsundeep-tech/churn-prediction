import os
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, plot_metric, plot_importance

from salary_prediction.model.abstract_model import AbstractPredictorModel, AbstractTrainerModel


class LGBMModelTrainer(AbstractTrainerModel):

    def train(self):
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.transformed_data[self.train_columns].astype('float'),
            self.transformed_data[self.target_column].astype('float'),
            test_size=0.20,
            random_state=31)
        model = self._train_model(X_train, X_valid, y_train, y_valid)
        self.plot(model)
        self.explain(model)
        return model

    def _train_model(self, X_train, X_valid, y_train, y_valid):
        lgbm = LGBMRegressor(objective='rmse',
                             seed=31,
                             n_estimators=100,
                             importance_type='gain')
        lgbm.fit(X_train,
                 y_train,
                 eval_set=[(X_train, y_train), (X_valid, y_valid)])

        return lgbm

    def explain(self, model):
        ax = plot_importance(model, ignore_zero=False, figsize=(40, 40))
        fig = ax.get_figure()
        fig_path = os.path.join(self.save_path, 'feature_importance.png')
        fig.savefig(fig_path)

    def plot(self, model):
        ax = plot_metric(model)
        fig = ax.get_figure()
        fig_path = os.path.join(self.save_path,
                                'train_val_rmse_comparison.png')
        fig.savefig(fig_path)


class LGBMModelPredictor(AbstractPredictorModel):

    def predict(self):
        predictions = self.model.predict(
            self.transformed_data[self.prediction_columns].astype('float'))
        self.transformed_data['predictions'] = predictions
        return self.transformed_data
