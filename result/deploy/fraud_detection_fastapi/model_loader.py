import pickle
import numpy as np
import xgboost as xgb


def load_model(model_path):
    with open ( model_path, 'rb' ) as f:
        model = pickle.load ( f )

    # Check if the loaded object has a predict method (indicating it's an XGBoost model)
    if not hasattr ( model, 'predict' ):
        raise ValueError (
            "The loaded model does not have a predict method. Please ensure the model is a valid XGBoost model." )

    return model


def make_prediction(model, data):
    """
    Make predictions using the provided model and input data.
    :param model: The trained model.
    :param data: The input data for prediction.
    :return: The predictions.
    """
    # Convert data to DMatrix if it's not already in that format
    if not isinstance ( data, xgb.DMatrix ):
        data = xgb.DMatrix ( data )

    # Make predictions
    predictions = model.predict ( data )

    return predictions
