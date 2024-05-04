import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess_data(data):
    # Perform the same preprocessing steps as before
    data['order date'] = pd.to_datetime ( data['order date (DateOrders)'] )
    data['shipping date'] = pd.to_datetime ( data['shipping date (DateOrders)'] )

    data['order year'] = data['order date'].dt.year
    data['order month'] = data['order date'].dt.month
    data['order day'] = data['order date'].dt.day
    data['order hour'] = data['order date'].dt.hour
    data['order minute'] = data['order date'].dt.minute

    data['shipping year'] = data['shipping date'].dt.year
    data['shipping month'] = data['shipping date'].dt.month
    data['shipping day'] = data['shipping date'].dt.day
    data['shipping hour'] = data['shipping date'].dt.hour
    data['shipping minute'] = data['shipping date'].dt.minute

    data_n = data.loc[:,
             ['Type', 'Days for shipment (scheduled)', 'order year', 'order month', 'order day', 'order hour',
              'order minute', 'Benefit per order', 'Category Name', 'Latitude', 'Longitude', 'Customer Segment',
              'Department Name', 'Market', 'Order City', 'Order Country', 'Order Item Discount',
              'Order Item Product Price', 'Order Item Quantity', 'Order Item Total', 'Order State', 'Product Name',
              'shipping year', 'shipping month', 'shipping day', 'shipping hour', 'shipping minute', 'Shipping Mode',
              'Late_delivery_risk', 'Order Status']]
    data_n['Order Status'] = [0 if i != 'SUSPECTED_FRAUD' else 1 for i in data_n['Order Status']]

    enc = LabelEncoder ()
    for col in data_n.columns:
        if data_n[col].dtype == 'object':
            data_n[col] = enc.fit_transform ( data_n[col] )

    y = data_n['Order Status']
    X = data_n.drop ( ['Order Status'], axis=1 )

    name = X.columns
    X = StandardScaler ().fit_transform ( X )
    X = pd.DataFrame ( X, columns=name )

    # No need to split the data here

    smote = SMOTE ( random_state=42 )
    X_resampled, y_resampled = smote.fit_resample ( X, y )

    return X_resampled, y_resampled
