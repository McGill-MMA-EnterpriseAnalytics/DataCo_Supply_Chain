import pandas as pd
import pickle
from sklearn.datasets import load_iris

# Load the model from a file
with open( 'iris_model.pkl', 'rb' ) as model_file:
    model = pickle.load(model_file)

# Load some sample data for inference (replace with your own data loading mechanism)
iris = load_iris()
X, y = iris.data, iris.target

# Perform inference
predictions = model.predict(X)

# Output the predictions (you could also write them to a file)
print(predictions)

# predictions_df = pd.DataFrame(predictions, columns=['predictions'])
# predictions_df.to_csv('predictions.csv', index=False)

