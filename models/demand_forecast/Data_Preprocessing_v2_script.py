#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
import mlflow
import mlflow.h2o
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

# Custom transformer for reading data
class DataReader(BaseEstimator, TransformerMixin):
    def __init__(self, filename):
        self.filename = filename
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.read_csv(self.filename, encoding='ISO-8859-1')


# Custom transformer for cleaning data
class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_clean = X.drop(columns=['Days for shipping (real)', 'Delivery Status', 'Late_delivery_risk', 
                                  'shipping date (DateOrders)', 'Benefit per order', 'Sales per customer', 'Category Id',
                                  'Order Profit Per Order', 'Order Item Discount', 'Order Item Total', 'Order Status', 
                                  'Customer Email', 'Customer Password', 'Latitude', 'Longitude', 'Product Description', 'Product Image',
                                  'Customer Fname', 'Customer Id', 'Customer Lname', 'Department Id',
                                  'Order Customer Id', 'Order Item Cardprod Id', 'Order Item Id',
                                  'Product Card Id', 'Product Category Id', 'Order Id', 'Customer Street',
                                  'Customer Zipcode', 'Order Zipcode', 'Order Item Product Price',
                                  'Product Price', 'Order Item Profit Ratio', 'Product Status'])
        X_clean['order date (DateOrders)'] = pd.to_datetime(X_clean['order date (DateOrders)'])
        X_clean['Year'] = X_clean['order date (DateOrders)'].dt.year
        X_clean['Month'] = X_clean['order date (DateOrders)'].dt.month
        X_clean.sort_values(by='order date (DateOrders)', inplace=True)
        X_clean.drop(columns=['order date (DateOrders)'], inplace=True)
        return X_clean

# Define the data pipeline
data_pipeline = Pipeline([
    ('data_reader', DataReader(filename="data/Q1_2015.csv")),
    ('data_cleaner', DataCleaner())
])


# In[8]:


transformed_data = data_pipeline.transform(X=None)


# In[9]:


transformed_data.head()

