import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

def main():
    # CSS to inject specified styling
    css_style = """
    <style>
        .reportview-container {
            background-color: #1F2F57;
            color: #FFFFFF;
        }
        h1, h2, h3 {
            color: #4DB6AC;
        }
        h1:hover, h2:hover, h3:hover {
            color: #25396E; /* Dark blue color on hover */
        }
        .stButton>button {
            background-color: #4DB6AC;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #25396E; /* Darker background on button hover */
            color: #FFFFFF;
        }
        .css-1d391kg {
            background-color: #25396E;
        }
    </style>
    """

    st.markdown(css_style, unsafe_allow_html=True)

    # Load models safely with error handling
    try:
        model1 = pickle.load(open('../../result/deploy/demand_forecast/demand_forecast.pkl', 'rb'))
        model2 = pickle.load(open('../../result/deploy/fraud_detection/fraud_detection_xgb.pkl', 'rb'))
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    def predict(input_data, model):
        dmatrix_data = xgb.DMatrix(input_data)
        predictions = model.predict(dmatrix_data)
        return predictions

    st.title('Supply Chain Prediction App')

    # Dropdown to select the model
    model_option = st.selectbox(
        'Select Model',
        ('Demand Forecast', 'Fraud Detection')
    )

    # File uploader allows user to add their own CSV
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            data = data.iloc[:, :-1]  # Assuming the last column is to be removed
            st.write(data)  # Display the uploaded data
        except Exception as e:
            st.error(f"Error reading data: {e}")
            st.stop()

    if st.button('Predict') and uploaded_file is not None:
        selected_model = model1 if model_option == 'Demand Forecast' else model2
        predictions = predict(data, selected_model)
        st.write(predictions)  # Display the predictions

if __name__ == "__main__":
    main()
