import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class DateConverter ( BaseEstimator, TransformerMixin ):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy ()
        X['order date'] = pd.to_datetime ( X['order date (DateOrders)'] )
        X['shipping date'] = pd.to_datetime ( X['shipping date (DateOrders)'] )
        return X


class FeatureEngineering ( BaseEstimator, TransformerMixin ):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy ()
        for feature in ['order date', 'shipping date']:
            prefix = feature.split ()[0]
            X[f'{prefix} year'] = X[feature].dt.year
            X[f'{prefix} month'] = X[feature].dt.month
            X[f'{prefix} day'] = X[feature].dt.day
            X[f'{prefix} hour'] = X[feature].dt.hour
            X[f'{prefix} minute'] = X[feature].dt.minute
        return X[['Type', 'order day', 'order hour', 'order minute', 'Benefit per order', 'Latitude', 'Longitude',
                  'Order City',
                  'Order Country', 'Order State', 'shipping day', 'shipping hour', 'shipping minute', 'Shipping Mode',
                  'Late_delivery_risk']]


class CategoricalEncoder ( BaseEstimator, TransformerMixin ):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder ()
                le.fit ( X[col] )
                self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy ()
        for col, encoder in self.encoders.items ():
            if col in X.columns:
                X[col] = encoder.transform ( X[col] )
        return X


class DataFrameConverter ( BaseEstimator, TransformerMixin ):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame ( X, columns=self.column_names )


def build_pipeline():
    all_cols = ['Type', 'order day', 'order hour', 'order minute', 'Benefit per order', 'Latitude', 'Longitude',
                'Order City',
                'Order Country', 'Order State', 'shipping day', 'shipping hour', 'shipping minute', 'Shipping Mode',
                'Late_delivery_risk']

    pipeline = ImbPipeline ( steps=[
        ('date_converter', DateConverter ()),
        ('feature_engineering', FeatureEngineering ()),
        ('encode_categorical', CategoricalEncoder ()),
        ('scaler', StandardScaler ()),
        ('to_dataframe', DataFrameConverter ( column_names=all_cols ))
    ] )

    return pipeline


def preprocess_for_prediction(df):
    pipeline = build_pipeline ()
    processed_data = pipeline.fit_transform ( df )
    y = df['Order Status'].map ( lambda x: 0 if x != 'SUSPECTED_FRAUD' else 1 )
    return processed_data
