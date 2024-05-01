import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Load model (make sure to provide the correct path to the model file)
model1 = pickle.load(open('../result/deploy/fraud_detection_xgb.pkl', 'rb'))
model2 = pickle.load(open('../result/deploy/fraud_detection_xgb.pkl', 'rb'))

def predict(input_data):
    # Convert the DataFrame to DMatrix
    dmatrix_data = xgb.DMatrix(input_data)
    # Make predictions
    predictions = model.predict(dmatrix_data)
    return predictions

st.title('Fraud Detection Model App')

# Dropdown to select the model
model_option = st.selectbox(
    'Select Model',
    ('Demand Forecast', 'Fraud Detection')
)

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='iso-8859-1')
    st.write(data)  # Display the uploaded data
    
    if st.button('Predict'):
        # Determine which model to use based on selection
        if model_option == 'Model 1':
            selected_model = model1
        else:
            selected_model = model2
        
        # Make predictions
        predictions = predict(data, selected_model)
        st.write(predictions)  # Display the predictions
