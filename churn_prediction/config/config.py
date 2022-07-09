



class ModelConfig:
    COLUMNS = {
        'numerical_features': ['yearsExperience', 'milesFromMetropolis'],
        'categorical_features': ['jobType', 'degree', 'major', 'industry'],
        'textual_features': [],
        "cardinality_features": ['companyId'],
        'target_feature': 'salary',
        'key_column': 'jobId'
    }
    MODEL_TYPE = 'catboost'
    TRAIN_FEATURES = [
        'companyId', 'jobType', 'degree', 'major', 'industry',
        'yearsExperience', 'milesFromMetropolis'
    ]
