import pickle

# Try to load the model locally to check its integrity
try:
    with open( 'iris_model.pkl', 'rb' ) as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load model:", e)
