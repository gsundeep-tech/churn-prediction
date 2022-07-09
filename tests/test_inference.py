import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import tempfile
from predict import predict_data

resources_path = os.path.join(os.path.dirname(__file__), 'resources')


class TestInference:

    def test_inference(self):
        with tempfile.TemporaryDirectory() as save_dir:
            predict_data(test_data_path=os.path.join(
                resources_path, 'train_features_sample.csv'),
                         model_path=os.path.join(resources_path, 'model.pkl'),
                         save_path=save_dir)
            predictions_path = os.path.join(save_dir, 'predictions.csv')
            assert os.path.isfile(predictions_path) is True