from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

# Define a request body model
class Item(BaseModel):
    features: list  # Adjust this based on the input your model expects

# Initialize the FastAPI app
app = FastAPI()

# Load your trained model
with open('fraud_detection_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict/")
def predict(item: Item):
    try:
        # Convert features to the appropriate format if necessary
        prediction = model.predict([item.features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello": "World"}
