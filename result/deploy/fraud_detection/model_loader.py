import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_prediction(model, data):
    try:
        prediction = model.predict(data)
        return prediction.tolist()
    except Exception as e:
        return str(e)
