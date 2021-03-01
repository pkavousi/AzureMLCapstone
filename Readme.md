# AzureML Engineering Capstone Project: Bank Customers Churn Model 

## Table of contents
   * [Overview](#Overview)
   * [Architectural Diagram](#Architectural-Diagram)
   * [Key Steps](#Key-Steps)
   * [AutoML](#AutoML)
   * [XGBoost](#XGBoost)
   * [Deployment](#Deployment)
   * [Screencast](#Screencast)
   * [Comments and future improvements](#Comments-and-future-improvements)
   * [Dataset Citation](#Dataset-Citation)
   * [References](#References)

***


## Overview

This project is formed by two parts:

- The first part consists of creating a machine learning production model using AutoML in Azure Machine Learning Studio. The best model of the AutoML is not deployed and just tested. The main focus of this project is to make an end to end machine learning model using custume preprocessing and models.
- The second part of the project uses XGBoos 1.4 which is downloaded as a ".whl" file. Moreover, the preprocessing step is packaged as a ".whl" file and used during training and deployment to fit_transform training data and just transform data during deployment. The preprocessing pipeline is flexible and can be expanded to more complex and custome preprocssing.

For both parts of the project I use the dataset that can be obtained from
from Kaggle [here](https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers). A bank is interested to know if a customer will churn given demographic and several other payment history data. The bank needs a system to predict customer who are likely to Churn. This projects investigates how such a system may be built by leveraging Bank history's data and Azure's Machine Learing platform.

The CSV file is uploaded in the datasets and imported here. The classification goal is to predict whether the customer will churn. The result of the prediction appears in _`Attrition_Flag
`_ and it is either _`Existing Customer`_ or _`Attrited Customer`_.

***
## Architectural Diagram

The architectural diagram is not very detailed by nature; its purpose is to give a rough overview of the operations. The diagram below is a visualization of the flow of operations from start to finish:

![Architectural Diagram](img/architechture.png?raw=true "Architectural Diagram") 


***
## Key Steps


The key steps of the project are described below:

- **Authentication and Data Import:**
A service principal ID and Password is necessary to run hyperparameter tuning notebook on your Azure account. You can replace yours inside "your password". Service Principal ID gives permission to the code to access resources based on the specifications that you assign to tenant when you create it. Details of Service Principal ID creation is not discussed here. Further instructions can be found in this [link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication).

- **Automated ML Experiment:**
At this point, security is enabled and authentication is completed. This step involves the creation of an experiment using Automated ML, configuring a compute cluster, and using that cluster to run the experiment.

- **Hyperparameter tuning Experiment:**
A XGBoost(1.3.0) model is used with hyperdrive to deploy a webservice that can be used to predict churn probability and label. The model uses two preprocessing steps that uses Categorical Encoder and Standard Scaler on training data and later on new data during deployment. The preprocessing steps are packages as a ".whl" in the bin folder.

- **Deploy the Best Model:**
The best model of hyperparameter optimization experiment with custom preprocessing and XGBoost is deployed locally to ensure it works and also debug its errors. Subsequently, a webservice is deployed that can respond to json format requests.

- **Logging:**
Logging is ran for the deployed XGBoost model.

- **Documentation:**
The documentation includes: 1. the [screencast](https://youtu.be/UAcjcypK0ro) that shows the entire process of the working ML application; and 2. this README file that describes the project and documents the main steps.

***

## AutoML

### **AutoML Experiment**

The AutoML is configured to run on a compute target. The data cleaning and preprocessing is left to the AutoML. Thus the featurization parameter is set to auto. Since no validation dataset is passed into the AutoML, cross validation on the test data is allowed and n_cross_validations is set to 5. To enable AutoML stop poorly performing runs, enable_early_stopping is set to True. Together with experiment_timeout_minutes which is set to 15, both parameters help to conserve resources (time and compute) available for experimentation.

The experiment runs for about 15 min. and is completed:

![AutoML_run](img/AutoML_run.PNG?raw=true "AutoML Completed")

The completed AutoML with their corresponding Run IDs are shown here:

![AutoML_run_details](img/amldetails.PNG?raw=true "AutoML Details")



### **AutoML Results**

The next step in the procedure is to retrive the best model of automl.
- It is identified that the best model is a LightGBMClassifier with a AUC of 0.9917.
    - Investigating the pattern of missing values and exploring more sophisticated ways to fill them
    - Dedicated hyperparameter tuning to further improve model performance
- An examination of model pipeline that the experiment outputted showed that no data transformation was done as shown below:


```
datatransformer
{'enable_dnn': None,
 'enable_feature_sweeping': None,
 'feature_sweeping_config': None,
 'feature_sweeping_timeout': None,
 'featurization_config': None,
 'force_text_dnn': None,
 'is_cross_validation': None,
 'is_onnx_compatible': None,
 'logger': None,
 'observer': None,
 'task': None,
 'working_dir': None}

MaxAbsScaler
{'copy': True}
```
- The best model is as following:

```
LightGBMClassifier
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'n_jobs': 1,
 'num_leaves': 31,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}
```
The best model of AutoML is registered to be ready for deployment. However, please note that this model is not deployed in this notebook.

![AutoML_run_registered](img/AutoML_run_registered.PNG?raw=true "AutoML_run_registered")

### **AutoML test on test data**

Part of the data was kept separate from AutoML experiment and its crossvalidation to evaluate AutoML model on new data. Results show that model is performin good and TypeI and TypeII erros are not significant. False Negative(FN) are slightly less that False Positive(FP). Usually Banks care more for False Negatives or TypeII errors:

![Confusion Matrix](img/confusionmatrix.PNG?raw=true "Confusion Matrix")


## XGBoost

### **XGBoost Model Training and Optimization**

The aim of this HyperDrive experiment is to use a XGBoost Classifier from a ".whl" file and optimize its hyperparameters.A sklearn Pipeline consisting of Categorical Encoder and a Standard Scaler is fit to the training data and then transformed it for training step. The scaler is dumped to be used just as a transformer during deployment.

```
churn_pipe = Pipeline(
    [
        ('categorical_encoder',
            CategoricalEncoder(variables=vars_cat)),
        ('scaler', StandardScaler()),
    ]
)
```
The main Hyperparameters are used are `gamma` that can control overfitting, `max_depth`, `learning_rate`,`colsample_bytree` and `scale_pos_weight`, which can help to aleviate problem with imbalanced datasets. A Bayesian hyperparameter optimization approach is used in this network. Thus, termination policy is not necessary. The top two optimized hyperparameter sets are as follwoing:

![HPO completed](img/HPO_run.PNG?raw=true "Hyperparameter optimization completed")
![HPO](img/HPO.PNG?raw=true "Hyperparameter Optimization") 

Moreover, we can also see the details of the best model in AzureML portal as following:

![HPO_completedRun](img/HPO_completedRun.PNG?raw=true "Hyperparameter Optimization Completion") 

The AUC score of the best model is 0.9936 which is slightly higher than AutoML best AUC of 0.9917. However, the purpose here is not to compare the models since both have room for improvements.
The confusion matrix of the best XGBoost of hyperdrive is: 

![Confusion Matrix](img/confusionmatrix-HPO.PNG?raw=true "Confusion Matrix")

## Deployment

### **Local Docker deployement**

It is usually easier to debug your model locally before deploying it as a Webservise Endpoint. This helps debuuging the model especially if there are preprocessing steps involved. 

![HPO](img/local_docker.PNG?raw=true "local deployment")

### **Publish and Consume a Endpoint**
A number of things are important to deploying the trained XGBoost model. They include:
- Optimal hyperparameters obtained for the classifiers through the hyperdrive Steps.
- XGBoost(1.3.0) is uploaded as '.whl' file and added to the environemt for training and deployment.
- Categorical Encoder class is packaged into ".whl" file and added to the specified environemt(`myenv`) for both training and deployment. This ensures that SKlearn versions and incompatibility between packages does not invalidate the `score.py` functionality.
```
whl_url="https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-1.3.0_SNAPSHOT%2Bb3193052b31e3e984f0f3d9c2c67ae3c2e114f9b-py3-none-manylinux2010_x86_64.whl"

# The custome preprocessing package is uploaded to 
whl_url2 = Environment.add_private_pip_wheel(workspace=ws,
                                            file_path = "bin/prep_package-0.1.0-py3-none-any.whl",exist_ok=True)

# We add the packages that are necessaary
conda_dep = CondaDependencies()
conda_dep.add_pip_package(whl_url)
conda_dep.add_pip_package(whl_url2)
conda_dep.add_pip_package("numpy~=1.18.0")
conda_dep.add_pip_package("scikit-learn==0.22.1")
conda_dep.add_pip_package("pandas~=0.25.0")
# We add all the added packages to myenv
myenv.python.conda_dependencies=conda_dep
```
- The built preprocessing package is named 'prep' and it has only one class as CategoricalEncoder which is based on SKlearn custome prewprocessing class
```
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = pd.concat([X,
                       pd.get_dummies(X[self.variables], drop_first=True)],
                       axis=1)

        X.drop(labels=self.variables, axis=1, inplace=True)

        # add missing dummies if any
        missing_vars = [var for var in self.dummies if var not in X.columns]

        if len(missing_vars) != 0:
            for var in missing_vars:
                X[var] = 0

        return X
```
- The prep package is use with the following command in the `train.py` and `score.py`.

```
from prep import CategoricalEncoder
```
These three ensure that the test data undergoes the same preprocessing that the train data did and that it contains only the columns the `XGBoost` was trained on. In addition, the following are necessary for model deployment:
- `score.py` which details how the deployed model interacts with requests
- An inference configuration which specifies the environment into which the model is deployed
- A deployment configuration specifying the resource allocation to the deployment model and additional characteristics such as `Application Insights`

The Best model is deployed using Azure Container Instances (ACI) to provision compute for demanding workloads. The deployed `uri` can be invoked by sending a Json file.
![endpoint in portal](img/deploy_endpoint.PNG?raw=true "deployed endpoint in portal")
![endpoint](img/deployeduri.PNG?raw=true "webservice deployment")

***
## Screencast

The screen recording can be found [here](https://youtu.be/UAcjcypK0ro) and it shows the project in action. More specifically, the screencast demonstrates:

* The working deployed ML model endpoint
* The deployed model demo
* Successful API requests to the endpoint with a JSON payload

***
## Comments and future improvements

* An ensemble model and additional feature engineering could improve the model performance. This requires addditional preprocessing steps in the pipeline. 
* Giving the hyperdrive more time and iterations could significantly improve the model performance. However, this comes with additional cost and time.
* Although data imbalanced is partly addressed by `scale_pos_weight` in XGBoost, a SMOTE preprocessing step in the pipeline should be further tested.

***
## Dataset Citation

https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers

***
## References

- Udacity Nanodegree material
- [Imbalanced Data : How to handle Imbalanced Classification Problems](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/)
- [Consume an Azure Machine Learning model deployed as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)
