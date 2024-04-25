from flask import Flask, request, jsonify
import h2o
import os

app = Flask(__name__)

# Initialize the H2O server
h2o.init()

# Load the model from the specified path
model_path = os.getenv('MODEL_PATH', '/app/DeepLearning_1_AutoML_8_20240423_184508')
model = h2o.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    hf = h2o.H2OFrame(input_data)
    predictions = model.predict(hf)
    return jsonify(predictions.as_data_frame().to_dict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
