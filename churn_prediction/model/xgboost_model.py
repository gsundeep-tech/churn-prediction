import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from numpy import sqrt
from numpy import argmax
from churn_prediction.config.config import ModelConfig

from churn_prediction.model.abstract_model import AbstractPredictorModel, AbstractTrainerModel


class XGBoostModelTrainer(AbstractTrainerModel):
    def train(self, data, train_columns, target_column, model_key):
        X_train, X_valid, y_train, y_valid = train_test_split(
            data[train_columns].astype('float'),
            data[target_column].astype('float'),
            test_size=0.20,
            random_state=31)
        model = self._train_model(X_train, X_valid, y_train, y_valid)
        f1_threshold, recall_treshold = self.calculate_thresholds(model, X_valid, y_valid, model_key)
        self.plot(model, model_key)
        self.explain(model, model_key)
        return model, f1_threshold, recall_treshold

    def calculate_thresholds(self, model, X_valid, y_valid, model_key):
        probs = model.predict_proba(X_valid)
        preds = probs[:, 1]
        recall_treshold = self._calculate_roc_auc_threshold(preds, y_valid, model_key)
        f1_threshold = self._calculate_f1_score_threshold(preds, y_valid, model_key)
        return f1_threshold, recall_treshold
    
    @staticmethod
    def _calculate_pos_weight_ratio(y_labels):
        ratio = sum(y_labels == 0) // sum(y_labels == 1)
        return ratio

    def _calculate_roc_auc_threshold(self, preds, y_valid, model_key):
        fpr, tpr, threshold = roc_curve(y_valid, preds)
        gmeans = sqrt(tpr * (1 - fpr))
        ix = argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, label='XGboost')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        path = os.path.join(self.save_path, f'{model_key}_roc_auc_curve.png')
        plt.savefig(path)
        plt.clf()
        plt.close()
        return threshold[ix]

    def _calculate_f1_score_threshold(self, preds, y_valid, model_key):
        precision, recall, thresholds = precision_recall_curve(y_valid, preds)
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        fscore = np.nan_to_num(fscore, nan=0)
        ix = argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        # plot the roc curve for the model
        no_skill = len(y_valid[y_valid==1]) / len(y_valid)
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, label='XGBoost')
        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        # show the plot
        path = os.path.join(self.save_path, f'{model_key}_precision_recall_curve.png')
        plt.savefig(path)
        plt.clf()
        plt.close()
        return thresholds[ix]

    def _train_model(self, X_train, X_valid, y_train, y_valid):
        pos_weight_ratio = self._calculate_pos_weight_ratio(y_train)

        parameters = {
            'min_child_weight': [1, 5],
            'gamma': [0.5, 1, 1.5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'learning_rate': [0.1, 0.2, 0.3],
        }

        # parameters = {
        #     'min_child_weight': [1, 5],
        #     #'gamma': [0.5, 1, 1.5],
        #     #'subsample': [0.6, 0.8, 1.0],
        #     #'colsample_bytree': [0.6, 0.8, 1.0],
        #     # 'learning_rate': [0.01, 0.1, 0.5],
        #     'learning_rate': [0.1, 0.2, 0.3],
        #     'subsample': [0.5, 0.9],
        #     'colsample_bytree': [0.6, 0.8, 1.0],
        # }

        clf = XGBClassifier(scale_pos_weight=pos_weight_ratio,
                            objective='binary:logistic',
                            early_stopping_rounds=5,
                            n_estimators=1000,
                            eval_metric=["error", "logloss"], max_depth=5)

        search = GridSearchCV(estimator=clf, param_grid=parameters,
                              scoring='roc_auc', n_jobs=10, cv=3, verbose=2)
        # scoring='recall' , 
        # search = RandomizedSearchCV(estimator=clf, param_distributions=parameters,
        #                             scoring='roc_auc', n_jobs=10, cv=3, verbose=2)

        search.fit(X_train,
                   y_train,
                   eval_set=[(X_train, y_train), (X_valid, y_valid)])

        return search.best_estimator_

    def explain(self, model, model_key):
        ax = plot_importance(model)
        fig = ax.get_figure()
        fig_path = os.path.join(self.save_path, f'{model_key}_feature_importance.png')
        fig.savefig(fig_path)
        plt.close(fig)

    def plot(self, model, model_key):
        # ax = plot_metric(model)

        results = model.evals_result()

        plt.figure(figsize=(10, 7))
        plt.plot(results["validation_0"]["logloss"], label="Training loss")
        plt.plot(results["validation_1"]["logloss"], label="Validation loss")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        path = os.path.join(self.save_path, f'{model_key}_loss.png')
        plt.savefig(path)
        plt.clf()
        plt.close()

        plt.figure(figsize=(10, 7))
        plt.plot(results["validation_0"]["error"], label="Training Error")
        plt.plot(results["validation_1"]["error"], label="Validation Error")
        plt.xlabel("Number of trees")
        plt.ylabel("Error")
        plt.legend()
        path = os.path.join(self.save_path, f'{model_key}_error.png')
        plt.savefig(path)
        plt.clf()
        plt.close()

class XGBoostModelPredictor(AbstractPredictorModel):

    @staticmethod
    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    def predict(self, model, data, prediction_columns, f1_threshold, recall_threshold):
        if ModelConfig.PREDICT_MAX_PRECISION:
            threshold = f1_threshold
        else:
            threshold = recall_threshold
        probabilities = model.predict_proba(data[prediction_columns].astype('float'))
        probabilities_ones = probabilities[:, 1]
        predictions = self.to_labels(probabilities_ones, threshold)
        data['probabilities'] = probabilities_ones
        data['predictions'] = predictions
        return data
