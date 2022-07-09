import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import tempfile
from train import train_model

resources_path = os.path.join(os.path.dirname(__file__), 'resources')


class TestTraining:

    def test_training(self):
        with tempfile.TemporaryDirectory() as save_dir:
            train_model(training_data_features_path=os.path.join(
                resources_path, 'train_features_sample.csv'),
                        training_data_targets_path=os.path.join(
                            resources_path, 'train_salaries_sample.csv'),
                        model_save_path=save_dir)
            model_path = os.path.join(save_dir, 'model.pkl')
            assert os.path.isfile(model_path) is True
