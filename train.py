import os
import argparse
from churn_prediction.training_pipeline import TrainingPipeline
from churn_prediction.data.ingestor import DataIngestor
from churn_prediction.utils.file_utils import generate_save_path


def train_model(training_data_path, training_metadata_path,
                model_save_path):

    ingestor = DataIngestor(data_features_path=training_data_path,
                            data_targets_path=training_metadata_path)
    data_features, data_targets = ingestor.load_data()

    pipeline = TrainingPipeline(data_features=data_features,
                                data_targets=data_targets,
                                model_save_path=model_save_path)
    pipeline.train()


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline Arguments")
    parser.add_argument('--training_data_path',
                        type=str,
                        required=True,
                        help="Path of the training data")
    parser.add_argument('--training_metadata_path',
                        type=str,
                        required=True,
                        help="Path of the training metadata")
    parser.add_argument('--model_save_path',
                        type=str,
                        required=False,
                        help="Path to save the trained model")

    args = parser.parse_args()
    training_data_path = args.training_data_path
    training_metadata_path = args.training_metadata_path
    model_save_path = args.model_save_path

    if not model_save_path:
        model_save_path = generate_save_path()

    train_model(training_data_features_path=training_data_path,
                training_data_targets_path=training_metadata_path,
                model_save_path=model_save_path)


if __name__ == "__main__":
    main()
