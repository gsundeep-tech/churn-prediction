import argparse

from churn_prediction.prediction_pipeline import PredictionPipeline
from churn_prediction.data.ingestor import DataIngestor
from churn_prediction.utils.file_utils import generate_save_path


def predict_data(test_data_path, model_path, save_path):
    ingestor = DataIngestor(data_path=test_data_path)
    data = ingestor.load_data()

    predictor = PredictionPipeline(data=data,
                                   model_path=model_path,
                                   save_path=save_path)
    predictor.predict()


def main():
    parser = argparse.ArgumentParser(description="Prediction Arguments")
    parser.add_argument("--test_data_path",
                        required=True,
                        type=str,
                        help="Test data path")
    parser.add_argument("--model_path",
                        required=True,
                        type=str,
                        help="Model Path")
    parser.add_argument("--predictions_save_path",
                        required=False,
                        type=str,
                        help="Predictions saving path directory")
    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path
    predictions_save_path = args.predictions_save_path
    if not predictions_save_path:
        predictions_save_path = generate_save_path()

    predict_data(test_data_path=test_data_path,
                 model_path=model_path,
                 save_path=predictions_save_path)


if __name__ == "__main__":
    main()