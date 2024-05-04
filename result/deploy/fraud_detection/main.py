from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
from preprocessing_script import preprocess_for_prediction
from model_loader import load_model, make_prediction
import logging
import numpy as np
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Load the model
model = load_model('fraud_detection_xgb.pkl')

app = FastAPI()

@app.post("/upload_predict/")
async def upload_predict(file: UploadFile = File(...)):
    logger.info(f"Received file with filename: {file.filename}")
    try:
        # Read file content into DataFrame
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents), encoding='ISO-8859-1')
        logger.info(f"Data read successfully with columns: {df.columns.tolist()}")

        # Perform preprocessing for prediction
        processed_data = preprocess_for_prediction(df)

        # Make prediction using the model
        prediction = make_prediction(model, processed_data)

        # Convert prediction probabilities to binary class labels
        prediction_class = (prediction > 0.5607).astype(int)

        # Convert prediction class to list for JSON serialization
        prediction_list = prediction_class.tolist()

        return {"message": "Prediction made successfully", "prediction": prediction_list}
    except Exception as e:
        error_msg = f"An error occurred during processing: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
