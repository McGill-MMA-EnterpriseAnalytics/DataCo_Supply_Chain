from flask import Flask, jsonify
import h2o
import os

app = Flask( __name__ )

# Initialize the H2O server
h2o.init()

# Load the model from the specified path
model_path = os.getenv('MODEL_PATH', '/app/DeepLearning_1_AutoML_8_20240423_184508')
model = h2o.load_model(model_path)


def generate_predictions():
    input_data_path = '/app/DataCo_cleaned_part1.csv'

    input_data = h2o.import_file(path=input_data_path)

    predictions = model.predict(input_data)
    # Convert predictions to a pandas DataFrame and return
    return predictions.as_data_frame()


@app.route ( '/predict', methods=['GET'] )
def predict():
    # Generate predictions
    pred_df = generate_predictions ()

    # Save predictions to a CSV file
    pred_df.to_csv ('/app/predictions.csv', index=False)

    # Return JSON response with confirmation or serve the CSV directly
    return jsonify ({'message': 'Predictions generated and saved to /app/predictions.csv'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)