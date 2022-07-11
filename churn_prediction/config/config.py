class ModelConfig:
    COLUMNS = {
        'numerical_features': ['age', 'annual_premium', 'days_since_insured'],
        'categorical_features': [
            'gender', 'driving_license', 'previously_insured', 'vehicle_age',
            'vehicle_damage'
        ],
        'textual_features': [],
        "cardinality_features": ['policy_sales_channel', 'region_code'],
        'target_feature': 'response',
        'key_column': 'cust_id'
    }
    MODEL_TYPE = 'xgboost'
    TRAIN_FEATURES = [
        'age', 'annual_premium', 'days_since_insured', 'gender', 'driving_license', 'previously_insured', 'vehicle_age',
            'vehicle_damage', 'policy_sales_channel'
    ]
    USE_REGION_CODE_MODEL = True
    CARDINALITY_FEATURES_REGION_BASED_MODEL = ['policy_sales_channel']
    MIN_DATA_POINTS = 15000
    MODEL_KEY = 'region_code'
    PREDICT_MAX_PRECISION = True