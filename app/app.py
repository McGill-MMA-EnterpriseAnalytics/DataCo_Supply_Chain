import streamlit as st
import pandas as pd
import pickle

# Load model (make sure to provide the correct path to the model file)
model = pickle.load(open('../result/deploy/fraud_detection_xgb.pkl', 'rb'))

def predict(input_data):
    # This function will make predictions based on the model and input data
    return model.predict(input_data)

st.title('Fraud Detection Model App')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)  # Display the uploaded data
    if st.button('Predict'):
        predictions = predict(data)
        st.write(predictions)  # Display the predictions