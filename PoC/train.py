import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to disk
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)
