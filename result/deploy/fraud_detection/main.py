from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from preprocessing_script import preprocess_data  # Import the preprocessing function
from model_loader import load_model, make_prediction

# Define a request body model
class Item(BaseModel):
    Type: int
    order_day: int
    order_hour: int
    order_minute: int
    Benefit_per_order: float
    Latitude: float
    Longitude: float
    Order_City: int
    Order_Country: int
    Order_State: int
    shipping_day: int
    shipping_hour: int
    shipping_minute: int
    Shipping_Mode: int
    Late_delivery_risk: int

# Initialize the FastAPI app
app = FastAPI()

# Load the model
model_path = 'fraud_detection_xgb.pkl'
model = load_model(model_path)

@app.post("/predict/")
def predict_order(item: Item):
    # Convert Pydantic model to dictionary, then to DataFrame
    data = item.dict()
    data_df = pd.DataFrame([data])  # Convert the dictionary to a DataFrame

    # Preprocess the data using the preprocessing script
    X, _ = preprocess_data(data_df)

    # Perform prediction using the trained model
    try:
        prediction = make_prediction(model, X)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"Hello": "World"}
