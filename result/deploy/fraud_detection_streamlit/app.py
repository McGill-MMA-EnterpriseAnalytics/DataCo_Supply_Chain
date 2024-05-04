import streamlit as st
import pandas as pd
import requests

st.title('Fraud Detection Prediction')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Choose a file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)  # Display the uploaded dataframe in the UI

    # When the user clicks the 'Predict' button, the prediction is triggered
    if st.button('Predict'):
        # Convert DataFrame to CSV and send as bytes
        buf = BytesIO()
        data.to_csv(buf, index=False, encoding='ISO-8859-1')
        buf.seek(0)

        url = 'http://localhost:8000/upload_predict/'  # URL of your FastAPI endpoint
        files = {'file': buf}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            st.success("Prediction successful!")
            st.json(response.json())  # Display the prediction results in the UI
        else:
            st.error(f"Failed to predict. Error: {response.json()}")




