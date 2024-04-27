# %%
import pandas as pd
import numpy as np

from pathlib import Path
DATA_PATH = Path() / "../../data/raw"
DATA_PATH.mkdir(parents=True,exist_ok=True)

def load_csv(filename, data_path=DATA_PATH,encoding='ISO-8859-1'):
    csv_path = data_path / filename
    return pd.read_csv(csv_path,encoding=encoding)

data = load_csv("Q1_2015.csv", encoding='ISO-8859-1')

# %%
# data = pd.read_csv("../data/DataCoSupplyChainDataset.csv",encoding='ISO-8859-1')

# %%
data.info()

# %%
# data = supplyChain.iloc[:2000,:]
# data

# %%
data['order date']= pd.to_datetime(data['order date (DateOrders)'])
data['shipping date']= pd.to_datetime(data['shipping date (DateOrders)'])
data['order year']=data['order date'].dt.year
data['order month']=data['order date'].dt.month
data['order day']=data['order date'].dt.day
data['order hour']=data['order date'].dt.hour
data['order minute']=data['order date'].dt.minute

data['shipping year']=data['shipping date'].dt.year
data['shipping month']=data['shipping date'].dt.month
data['shipping day']=data['shipping date'].dt.day
data['shipping hour']=data['shipping date'].dt.hour
data['shipping minute']=data['shipping date'].dt.minute

# %%
data_n=data.loc[:,['Type','Days for shipment (scheduled)','order year','order month','order day','order hour','order minute','Benefit per order','Category Name','Latitude','Longitude','Customer Segment','Department Name','Market','Order City','Order Country','Order Item Discount','Order Item Product Price','Order Item Quantity','Order Item Total','Order State','Product Name','shipping year','shipping month','shipping day','shipping hour','shipping minute','Shipping Mode','Late_delivery_risk','Order Status']]
data_n.info()

# %%
data_n['Order Status'].value_counts()

# %%
data_n['Order Status']= [0 if i!='SUSPECTED_FRAUD' else 1 for i in data_n['Order Status']]

# %%
data_n['Order Status'].value_counts()

# %%
from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()
for i in data_n.columns:
    if data_n[i].dtype=='object':
        data_n[i]=enc.fit_transform(data_n[i])

# %%
data_n.info()

# %%
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

y=data_n['Order Status']
X=data_n.drop(['Order Status'],axis=1)
name = X.columns
X=StandardScaler().fit_transform(X)
X = pd.DataFrame(X, columns=name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Apply SMOTE to generate synthetic samples to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# %%
y_resampled.value_counts()

# %% [markdown]
# #### Feature Selection

# %% [markdown]
# Choosing Between MLflow and Auto-Sklearn
# 
# Use MLflow if you want to track experiments, version models, and have a comprehensive view of your model's lifecycle.
# 
# Use Auto-Sklearn if you want to automate the process of model and hyperparameter selection based on the given dataset, especially when you are unsure about which models or parameters to use.
# 
# This is a simple feature selection using random forest thus TPOT was not necessary, but keep as the fact the it has been tested.

# %%
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
# from sklearn.model_selection import train_test_split

mlflow.set_experiment('Feature_Selection_with_RFE')

with mlflow.start_run():
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf_classifier, n_features_to_select=15, step=1)
    rfe.fit(X_train, y_train)
    
    # Transform the data
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    # Train a new classifier on the transformed data
    rf_classifier.fit(X_train_rfe, y_train)
    
    # Evaluate the model
    score = rf_classifier.score(X_test_rfe, y_test)
    print(f"Model score after RFE: {score:.4f}")
    
    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("n_features_to_select", 15)
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(rf_classifier, "model")
    
    # Get and log the ranking of the features
    ranking = rfe.ranking_
    print(f"Feature ranking: {ranking}")
    mlflow.log_param("feature_ranking", ranking.tolist())

# To view the experiments, run the MLflow UI in terminal:
# mlflow ui


# %%
# from tpot import TPOTClassifier

# tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
# tpot.fit(X_train, y_train)
# print(tpot.score(X_test, y_test))

# Export the best model pipeline
# tpot.export('tpot_classification_pipeline.py')

# %%
selected_features = name[rfe.support_]
print("Selected features:", selected_features.tolist())

# %%
X_resampled_sel = X_resampled[selected_features.tolist()]
X_test_sel =  X_test[selected_features.tolist()]

# %% [markdown]
# #### H2O

# %%
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# %%
# transform df into a compatible data format
hf_dataset = h2o.H2OFrame(data_n)

# %%
# ensure model will be on classification task
hf_dataset['Order Status'] = hf_dataset['Order Status'].asfactor()

# %%
# train test split
train, test = hf_dataset.split_frame(ratios=[0.75],seed=1)

y = 'Order Status'
X = hf_dataset.columns
X.remove(y)

# %%
h2o_aml = H2OAutoML(max_models = 12, seed = 1, exclude_algos = ["StackedEnsemble"], verbosity="info") #max_runtime_secs=120,

# %%
h2o_aml.train(x = X, y = y, training_frame = train)

# %%
# Retrieve the leaderboard
leaderboard = h2o_aml.leaderboard

# Extract model IDs
model_ids = leaderboard['model_id'].as_data_frame(use_pandas=True)['model_id']

# Loop through the models and print their confusion matrices
for model_id in model_ids:
    model = h2o.get_model(model_id)
    performance = model.model_performance(hf_dataset)
    print(f"Confusion Matrix for model {model_id}:")
    print(performance.confusion_matrix())

# %%
# Access the best model
best_model = h2o_aml.leader

# Print a detailed summary of the model
print(best_model)

# Performance on training data
performance_train = best_model.model_performance(hf_dataset)

# Performance on a test set (if you have split your data)
performance_test = best_model.model_performance(test)

# Variable importance
variable_importance = best_model.varimp(use_pandas=True)
print(variable_importance)

# ROC curve (for classification)
roc_curve = performance_train.roc()


# %%
# Save the model
# model_path = h2o.save_model(model=best_model, force=True) 

# %%
best_model.get_params()

# %%
params = {
    'max_depth': best_model.params['max_depth']['actual'],
    'learning_rate': best_model.params['learn_rate']['actual'],
    'n_estimators': best_model.params['ntrees']['actual'],
    'subsample': best_model.params['sample_rate']['actual'],
    'colsample_bytree': best_model.params['col_sample_rate']['actual']
}

# %%
import xgboost as xgb

# Convert your dataset to DMatrix object
dtrain = xgb.DMatrix(X_resampled_sel, label=y_resampled)

# Set up XGBoost parameters (make sure to convert parameter names)
xgb_params = {
    'max_depth': params['max_depth'],
    'eta': params['learning_rate'],
    'subsample': params['subsample'],
    'colsample_bytree': params['colsample_bytree'],
    'objective': 'binary:logistic'  # or 'reg:squarederror' depending on your task
}

# Train the XGBoost model
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=params['n_estimators'])


# %%
import pickle

with open('../../result/fraud_detect.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

# %% [markdown]
# ### SHAP

# %%
import shap

# Load JS visualization code to notebook
shap.initjs()

# retrieve model parameters
# best_xgb_model = grid_xgb.best_estimator_

# Explain the model's predictions using SHAP
explainer = shap.Explainer(xgb_model)

# Compute SHAP values
shap_values = explainer(X_resampled_sel)

shap.summary_plot(shap_values, X_resampled_sel)

# %%
# Load JS visualization code to the notebook
shap.initjs()

# Find indices where the label is 1
indices_label_1 = [i for i, label in enumerate(y_resampled) if label == 1]

instance_index = indices_label_1[0]  # Adjust this index to plot other instances
shap.force_plot(explainer.expected_value, shap_values.values[instance_index], feature_names=X_resampled_sel.columns)

# %%
indices_label_0 = [i for i, label in enumerate(y_resampled) if label == 0]

instance_index = indices_label_0[0]  # Adjust this index to plot other instances
shap.force_plot(explainer.expected_value, shap_values.values[instance_index], feature_names=X_resampled_sel.columns)

# %%
# Generate the summary plot
shap.summary_plot(shap_values, X_resampled_sel, plot_type="bar")


# %%



