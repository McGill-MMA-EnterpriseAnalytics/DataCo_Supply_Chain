import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pathlib import Path

DATA_PATH = Path() / "../../data/processed"
DATA_PATH.mkdir(parents=True,exist_ok=True)

def save_csv(data, filename, data_path=DATA_PATH, encoding='ISO-8859-1'):
    csv_path = data_path / filename
    data.to_csv(csv_path, index=False, encoding=encoding)

class DateConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['order date'] = pd.to_datetime(X['order date (DateOrders)'])
        X['shipping date'] = pd.to_datetime(X['shipping date (DateOrders)'])
        return X

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in ['order date', 'shipping date']:
            prefix = feature.split()[0]
            X[f'{prefix} year'] = X[feature].dt.year
            X[f'{prefix} month'] = X[feature].dt.month
            X[f'{prefix} day'] = X[feature].dt.day
            X[f'{prefix} hour'] = X[feature].dt.hour
            X[f'{prefix} minute'] = X[feature].dt.minute
        X_n = X.loc[:,['Type','Days for shipment (scheduled)','order year','order month','order day','order hour',
                        'order minute','Benefit per order','Category Name','Latitude','Longitude','Customer Segment',
                        'Department Name','Market','Order City','Order Country','Order Item Discount','Order Item Product Price',
                        'Order Item Quantity','Order Item Total','Order State','Product Name','shipping year','shipping month',
                        'shipping day','shipping hour','shipping minute','Shipping Mode','Late_delivery_risk','Order Status']]
        return X_n

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        
    def fit(self, X, y=None):
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders[col] = le
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, encoder in self.encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])
        return X
    
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.column_names)

def build_pipeline():
    date_cols = ['order date (DateOrders)', 'shipping date (DateOrders)']
    
    all_cols = ['Type','Days for shipment (scheduled)','order year','order month','order day','order hour',
                        'order minute','Benefit per order','Category Name','Latitude','Longitude','Customer Segment',
                        'Department Name','Market','Order City','Order Country','Order Item Discount','Order Item Product Price',
                        'Order Item Quantity','Order Item Total','Order State','Product Name','shipping year','shipping month',
                        'shipping day','shipping hour','shipping minute','Shipping Mode','Late_delivery_risk','Order Status']


    pipeline = ImbPipeline(steps=[
        ('date_converter', DateConverter()),
        ('feature_engineering', FeatureEngineering()),
        ('encode_categorical', CategoricalEncoder()),
        ('scaler', StandardScaler()),
        ('to_dataframe', DataFrameConverter(column_names=all_cols))  
    ])
    
    return pipeline

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def select_and_concatenate_datasets(X_resampled, y_resampled, X_test, y_test):
    selected_features = ['Type', 'order day', 'order hour', 'order minute', 'Benefit per order', 'Latitude', 'Longitude', 'Order City', 'Order Country', 'Order State', 'shipping day', 'shipping hour', 'shipping minute', 'Shipping Mode', 'Late_delivery_risk']
    X_resampled_sel = X_resampled[selected_features]
    X_test_sel = X_test[selected_features]
    train_resampled = pd.concat([X_resampled_sel, y_resampled], axis=1)
    test_sel = pd.concat([X_test_sel, y_test], axis=1)
    concatenated_datasets = pd.concat([train_resampled, test_sel])
    return concatenated_datasets

def process_data_from_csv(filepath):
    # Read CSV file
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    # Build and apply the preprocessing pipeline
    pipeline = build_pipeline()
    df_processed = pipeline.fit_transform(df)
    
    # Prepare target and features
    y = df['Order Status'].map(lambda x: 0 if x != 'SUSPECTED_FRAUD' else 1)
    X = df_processed.drop('Order Status', axis=1)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Apply SMOTE to the training data
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    
    # Concatenate datasets for the final dataset
    final_dataset = select_and_concatenate_datasets(X_resampled, y_resampled, X_test, y_test)
    
    return final_dataset


save_csv(process_data_from_csv("../../data/raw/Q1_2015.csv"), 'Q1_2015_fraud.csv')
save_csv(process_data_from_csv("../../data/raw/future_data.csv"), 'future_data_fraud.csv')
