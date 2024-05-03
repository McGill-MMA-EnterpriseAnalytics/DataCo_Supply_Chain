from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

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

# Load your trained model
with open('fraud_detection_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict/")
def predict(item: Item):
    # Convert Pydantic model to dictionary, then to DataFrame or directly use in prediction
    data = item.dict()
    data_df = pd.DataFrame([data])  # Convert the dictionary to a DataFrame

    # Predict using the model
    try:
        prediction = model.predict(data_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"Hello": "World"}
