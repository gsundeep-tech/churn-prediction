# churn-prediction

Motor Insurance Renewal Prediction

# To Run

1. Install the requirements using the command `pip install -r requirements.txt`
2. For model training, run the file [train.py](train.py) with the necessary arguments  
   `Example Command: python train.py --training_data_path ./data/train.csv --training_metadata_path ./data/metadata.xlsx`
3. For model inference, run the file [predict.py](predict.py) with the necessary argumets  
   `Example Command: python predict.py --test_data_path ./data/test.csv --model_path ./executions/11072022174134/model.pkl`

# To Run tests

1. Install the test requirements using the command `pip install -r test-requirements.txt`
2. run the command `pytest tests`
