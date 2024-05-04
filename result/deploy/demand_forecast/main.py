from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Define a request body model
class Item(BaseModel):
    Type: str
    Days_for_shipment_scheduled: int
    Category_Name: str
    Customer_City: str
    Customer_Country: str
    Customer_Segment: str
    Customer_State: str
    Department_Name: str
    Market: str
    Order_City: str
    Order_Country: str
    Order_Item_Discount_Rate: float
    Sales: float
    Order_Region: str
    Order_State: str
    Product_Name: str
    Shipping_Mode: str
    Day_of_Week: int
    Month: int
    Year: int
    Week_of_Year: int

# Initialize the FastAPI app
app = FastAPI()

# Load your trained model
with open( 'demand_forecast.pkl', 'rb' ) as f:
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
