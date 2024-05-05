import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder

# Using an emoji as an icon in the title
st.set_page_config ( page_title="Fraud Detection Dashboard", page_icon=":package:" )

st.title ( 'Fraud Detection Dashboard :mag:' )

# Adding a subtitle or note for the company
st.markdown ( """
Developed for **DataCo Global** to analyze fraud detection results.
""" )

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader ( "Choose a file", type=['csv'] )
if uploaded_file is not None:
    data = pd.read_csv ( uploaded_file, encoding='ISO-8859-1' )
    if "order date(DateOrders)" in data.columns:
        data['order date(DateOrders)'] = pd.to_datetime ( data['order date(DateOrders)'] )
        begin_date = data['order date(DateOrders)'].min ()
        end_date = data['order date(DateOrders)'].max ()
        st.write ( f"Data period from {begin_date.date ()} to {end_date.date ()}" )

    with st.spinner ( 'Analyzing data...' ):
        # Convert DataFrame to CSV and send as bytes
        buf = BytesIO ()
        data.to_csv ( buf, index=False, encoding='ISO-8859-1' )
        buf.seek ( 0 )

        url = 'https://mlproject-smbfygbzda-uc.a.run.app/docs#/default/upload_predict_upload_predict__post' # Google Cloud URL
        # url = 'http://localhost:8000/upload_predict/'  # Local URL of the FastAPI endpoint
        files = {'file': buf}
        response = requests.post ( url, files=files )

        if response.status_code == 200:
            predictions = pd.DataFrame ( response.json ()['prediction'], columns=['Fraud Prediction'] )
            predictions.replace ( {0: 'No', 1: 'Yes'}, inplace=True )
            results = pd.concat ( [data, predictions], axis=1 )

            # Data summary
            st.header ( "Data Summary" )
            st.write ( f"Total records analyzed: {len ( results )}" )
            st.write (
                f"Fraudulent transactions detected: {results['Fraud Prediction'].value_counts ().get ( 'Yes', 0 )}" )

            # Visualizations for top 5 Order City and Order Country
            top_cities = results[results['Fraud Prediction'] == 'Yes']['Order City'].value_counts ().head ( 5 )
            top_countries = results[results['Fraud Prediction'] == 'Yes']['Order Country'].value_counts ().head ( 5 )

            fig_city = px.bar ( top_cities, orientation='v', labels={'value': 'Count', 'index': 'City'},
                                title="Top 5 Fraud Order Cities" )
            fig_country = px.bar ( top_countries, orientation='v', labels={'value': 'Count', 'index': 'Country'},
                                   title="Top 5 Fraud Order Countries" )
            st.plotly_chart ( fig_city, use_container_width=True )
            st.plotly_chart ( fig_country, use_container_width=True )

            # Ag-Grid interactive table with fraud details
            st.header ( "Detailed View of Fraudulent Transactions" )
            fraud_cases = results[results['Fraud Prediction'] == 'Yes']
            gb = GridOptionsBuilder.from_dataframe ( fraud_cases )
            gb.configure_pagination ( paginationAutoPageSize=True )  # Automatic page size
            gb.configure_side_bar ()  # Enable side bar
            gb.configure_default_column ( groupable=True, value=True, enableRowGroup=True, aggFunc='sum',
                                          editable=True )
            gridOptions = gb.build ()

            grid_response = AgGrid (
                fraud_cases,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                allow_unsafe_jscode=True,  # Set it to True to allow jsfunction to be injected
                height=500,
                width='100%',
                theme='streamlit'  # Correct theme
            )

            # Download button for the fraud cases
            st.download_button (
                label="Download Fraud Data as CSV",
                data=fraud_cases.to_csv ().encode ( 'utf-8' ),
                file_name='fraud_cases.csv',
                mime='text/csv',
            )

        else:
            st.error ( "Failed to analyze data. Please check the file and try again." )
else:
    st.info ( "Please upload a CSV file to start the analysis." )
