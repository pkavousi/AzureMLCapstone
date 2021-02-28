import argparse
import os
import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Experiment
from azureml.core import Workspace, Dataset
from xgboost import XGBClassifier

from prep import CategoricalEncoder

from sklearn.preprocessing import StandardScaler
from azureml.core.authentication import ServicePrincipalAuthentication

# replace with your credentials
svc_pr = ServicePrincipalAuthentication(
    tenant_id="replace with yours",
    service_principal_id="replace with yours",
    service_principal_password="replace with yours")

ws = Workspace(subscription_id="",
               resource_group="azureML",
               workspace_name="Chun2",
               auth=svc_pr)
experiment_name = 'BanckChurner-HPO2'
experiment=Experiment(ws, experiment_name)

datastore=ws.get_default_datastore()
dataset=Dataset.Tabular.from_delimited_files(datastore.path('UI/02-28-2021_030313_UTC/BankChurners.csv'))
ds=dataset.to_pandas_dataframe()
############
target = 'Attrition_Flag'

vars_num = [c for c in ds.columns if ds[c].dtypes!='O' and c!=target]

vars_cat = [c for c in ds.columns if ds[c].dtypes=='O' and c!=target]
##############
"""make a pipeline of the preprocessing steps that are applied to the data before training. 
The same preprocessing shouldbe applied to new data that will be fed to the endpoint during deployment"""
churn_pipe = Pipeline(
    [
        ('categorical_encoder',
            CategoricalEncoder(variables=vars_cat)),
         
        ('scaler', StandardScaler()),
    ]
)


def clean_data():   
    """
    This function split the data into train-test and then applies a series of transformations
    that are designed in the pipeline. The fitted transformer will be saved to be used during
    the deployment.
    
    Args:
    output: x_train, x_test, y_train, y_test
    """
    
    x_train, x_test, y_train, y_test = train_test_split(
        ds.drop('Attrition_Flag', axis=1),
        ds['Attrition_Flag'],
        test_size=0.2,
        random_state=0)
    
    churn_pipe.fit(x_train, y_train)
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(churn_pipe, 'outputs/pipeline.pkl')
    x_train=churn_pipe.transform(x_train)
    x_test=churn_pipe.transform(x_test)
    return x_train, x_test, y_train, y_test

run = Run.get_context()


def main():
    """
    if tyrain.py is called, then this function and its embeded functions will be executed.
    It cleans the data, train a model, log the metrics and scores, and saves the fitted model
    """
    x_train, x_test, y_train, y_test=clean_data()
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--colsample_bytree', type=float, default=1, help="Subsample of columns")
    parser.add_argument('--scale_pos_weight',type=float, default=1, help='scale for minority class. Increasing could help in imbalanaced dataset')
    parser.add_argument('--learning_rate',type=float,default=0.1, help='learning rate')
    parser.add_argument('--gamma',type=float,default=0.1, help='gamma')
    parser.add_argument('--max_depth',type=int,default=6, help='max_depth')
    
    args = parser.parse_args()
    # Logging is necessary for hyperparameter optimization
    run.log("colsample_bytree:", np.float(args.colsample_bytree))
    run.log("scale_pos_weight:", np.float(args.scale_pos_weight))
    run.log("learning_rate:", np.float(args.learning_rate))
    run.log("gamma:", np.float(args.gamma))
    run.log("max_depth:", np.int(args.max_depth))
    
    model = XGBClassifier( learning_rate =args.learning_rate,
    max_depth=args.max_depth,
    min_child_weight=1,
    gamma=args.gamma,
    subsample=1,
    colsample_bytree=args.colsample_bytree,
    objective= 'binary:logistic',
    nthread=-1,
    seed=20,
    scale_pos_weight=args.scale_pos_weight                    
    ,eval_metric='auc').fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])
    run.log("AUC%", np.float(roc))
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/model.pkl')

if __name__ == '__main__':
    main()
