# AzureML Enginnering Capstone Project: Bank Customers Churn Model 

## Table of contents
   * [Overview](#Overview)
   * [Architectural Diagram](#Architectural-Diagram)
   * [Key Steps](#Key-Steps)
   * [Screenshots](#Screenshots)
   * [Screen Recording](#Screen-Recording)
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
The documentation includes: 1. the [screencast](https://youtu.be/0AKGw1YOcXw) that shows the entire process of the working ML application; and 2. this README file that describes the project and documents the main steps.

***

## Screenshots

### **Step 2: Automated ML Experiment**

The AutoML is configured to run on a compute target. The data cleaning and preprocessing is left to the AutoML. Thus the featurization parameter is set to auto. Since no validation dataset is passed into the AutoML, cross validation on the test data is allowed and n_cross_validations is set to 5. To enable AutoML stop poorly performing runs, enable_early_stopping is set to True. Together with experiment_timeout_minutes which is set to 15, both parameters help to conserve resources (time and compute) available for experimentation.

The experiment runs for about 15 min. and is completed:

![AutoML completed](img/AutoML_run.PNG?raw=true "AutoML completed")

### **Step 3: Results**

The next step in the procedure is to retrive the best model of automl.
- It is identified that the best model is a LightGBMClassifier with a AUC of 0.9917.
    - Investigating the pattern of missing values and exploring more sophisticated ways to fill them
    - Dedicated hyperparameter tuning to further improve model performance
- An examination of model pipeline that the experiment outputted showed that no data transformation was done as shown below:

'''
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
'''
- The best model is as following:
'''
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
 '''
 
### **Step 4: Enable Application Insights**

After the deployment of the best model, I can enable _Application Insights_ and be able to retrieve logs:

**"Application Insights" enabled in the Details tab of the endpoint**

!["Application Insights" enabled](img/Best_model2_Application_Insights_enabled.JPG?raw=true "'Application Insights' enabled")

Screenshot of the tab running "Application Insights":

!["Application Insights" graphs](img/Best_model2_Application_Insights_tab.JPG?raw=true "'Application Insights' graphs")

We can see _Failed requests_, _Server response time_, _Server requests_ & _Availability_ graphs in real time.


**Running logs.py script**

Although we can enable _Application Insights_ at deploy time with a check-box, it is useful to be able to run code that will enable it for us. For this reason, I run the _logs.py_ Python file, where I put in _name_ the name of the deployed model (_best-model2_) and I add the line `service.update(enable_app_insights=True)`: 

![Running logs.py script](img/Best_model2-logs_py_running.JPG?raw=true "Running logs.py script")


### **Step 5: Swagger Documentation**

**Swagger** is a set of open-source tools built around the OpenAPI Specification that can help us design, build, document and consume REST APIs. One of the major tools of Swagger is **Swagger UI**, which is used to generate interactive API documentation that lets the users try out the API calls directly in the browser.

In this step, I consume the deployed model using Swagger. Azure provides a _Swagger JSON file_ for deployed models. This file can be found in the _Endpoints_ section, in the deployed model there, which should be the first one on the list. I download this file and save it in the _Swagger_ folder.

I execute the files _swagger.sh_ and _serve.py_. What these two files do essentially is to download and run the latest Swagger container (_swagger.sh_), and start a Python server on port 9000 (_serve.py_).

![swagger.sh run](img/best_model2-Swagger1B.JPG?raw=true "swagger.sh run")

![swagger.sh run](img/best_model2-Swagger1A.JPG?raw=true "swagger.sh run")

In the Live Demo page of Swagger UI:

![Swagger UI](img/Swagger_LiveDemo1.JPG?raw=true "Swagger UI")

I click on Live Demo button and am transfered in a demo page with a sample server:

![Swagger UI](img/Swagger_LiveDemo2.JPG?raw=true "Swagger UI Live Demo")

I delete the address in the address bar pointed with the red arrow and replace it with: `http://localhost:9000/swagger.json`. After hitting _Explore_, Swagger UI generates interactive API documentation that lets us try out the API calls directly in the browser. 

![Swagger runs on localhost](img/61.JPG?raw=true "Swagger runs on localhost")

We can see below the HTTP API methods and responses for the model:

**Swagger runs on localhost - GET & POST/score endpoints**

![Swagger runs on localhost](img/best_model2-Swagger1.JPG?raw=true "Swagger runs on localhost")

![Swagger runs on localhost - GET endpoint](img/best_model2-Swagger2.JPG?raw=true "Swagger runs on localhost - GET endpoint")

![Swagger runs on localhost - POST/score endpoint](img/best_model2-Swagger3.JPG?raw=true "Swagger runs on localhost - POST/score endpoint")


### **Step 6: Consume Model Endpoints**

Once the best model is deployed, I consume its endpoint using the `endpoint.py` script provided where I replace the values of `scoring_uri` and `key` to match the corresponding values that appear in the _Consume_ tab of the endpoint: 

**Consume Model Endpoints: running endpoint.py**

![endpoint.py](img/best_model2_enpoint_py.JPG?raw=true "endpoint.py")

![run endpoint.py](img/best_model2_enpoint_py_run.JPG?raw=true "run endpoint.py")


### **Step 7: Create, Publish and Consume a Pipeline**

In this second part of the project, I use the Jupyter Notebook provided: `aml-pipelines-with-automated-machine-learning-step.ipynb`. The notebook is updated so as to have the same dataset, keys, URI, cluster, and model names that I created in the first part. 

The purpose of this step is to create, publish and consume a pipeline using the Azure Python SDK. We can see below the relevant screenshots: 

**The Pipelines section of Azure ML Studio**

![Pipeline has been created](img/Pipeline_has_been_created.JPG?raw=true "Pipeline has been created")

![Pipeline Endpoint](img/Pipeline_Endpoint.JPG?raw=true "Pipeline Endpoint")

**Bankmarketing dataset with the AutoML module** 

![Bankmarketing dataset with the AutoML module](img/Bankmarketing_Dataset+AutoML_module.JPG?raw=true "Bankmarketing dataset with the AutoML module")

**Published Pipeline Overview showing a REST endpoint and an ACTIVE status** 

![Published Pipeline Overview showing a REST endpoint and an ACTIVE status](img/41.JPG?raw=true "Published Pipeline Overview showing a REST endpoint and an ACTIVE status")

![Published Pipeline Overview showing a REST endpoint and an ACTIVE status](img/42.JPG?raw=true "Published Pipeline Overview showing a REST endpoint and an ACTIVE status")

**Jupyter Notebook: RunDetails Widget shows the step runs** 

![Jupyter Notebook: RunDetails Widget shows the step runs](img/RunDetailsWidget1.JPG?raw=true "Jupyter Notebook: RunDetails Widget shows the step runs")

![Jupyter Notebook: RunDetails Widget shows the step runs](img/RunDetailsWidget2.JPG?raw=true "Jupyter Notebook: RunDetails Widget shows the step runs")

**In ML Studio: Completed run** 

![In ML Studio](img/40.JPG?raw=true "In ML Studio")

![In ML Studio](img/50.JPG?raw=true "In ML Studio")

![In ML Studio](img/51.JPG?raw=true "In ML Studio")

***
## Screen Recording

The screen recording can be found [here](https://youtu.be/0AKGw1YOcXw) and it shows the project in action. More specifically, the screencast demonstrates:

* The working deployed ML model endpoint
* The deployed Pipeline
* Available AutoML Model
* Successful API requests to the endpoint with a JSON payload


***
## Comments and future improvements

* As I have pointed out in the 1st project as well, the data is **highly imbalanced**:

![Highly imbalanced data](img/Imbalanced_data_plot.png?raw=true "Highly imbalanced data")

Although AutoML normally takes into account this imbalance automatically, there should be more room to improve the model's accuracy in predicting the minority class. For example, we could use Random Under-Sampling of majority class, or Random Over-Sampling of minority class, or even try different algorithms.

A side note here: out of curiosity, I clicked the 'Data guardrails' tab (see screenshots above, step 3) and found many interesting observations done by Azure analysis. Unfortunately, I ran out of time and was not able to look into this with more detail. My remark here is that even though I can understand that there must be time contraints in our runs, this can impede our in depth learning because we miss the chance to browse around looking for the many extra but less important things; this is really a shame. As a suggestion, it would be interesting to create a virtual environment with everything running in simulation -thus running with no actual cost- where the learner could freely look around.

* Another factor that could improve the model is increasing the training time. This suggestion might be seen as a no-brainer, but it would also increase costs and there must always be a balance between minimum required accuracy and assigned budget.

* I could not help but wonder how more accurate would be the resulting model in case `Deep Learning` was used, as we were specifically instructed _NOT_ to enable it in the AutoML settings. While searching for more info, I found this very interesting article in Microsoft Docs: [Deep learning vs. machine learning in Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-deep-learning-vs-machine-learning). There it says that deep learning excels at identifying patterns in unstructured data such as images, sound, video, and text. In my understanding, it might be an overkill to use it in a classification problem like this.

* Lastly, a thing that could be taken into account is any future change(s) in the dataset that could impact the accuracy of the model. I do not have any experience on how this could be done in an automated way, but I am sure that a method exists and can be spotted if/when such a need arises.

***
## Dataset Citation

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

***
## References

- Udacity Nanodegree material
- [App](https://app.diagrams.net/) used for the creation of the Architectural Diagram
- [Prevent overfitting and imbalanced data with automated machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls)
- [Dr. Ware: Dealing with Imbalanced Data in AutoML](https://www.drware.com/dealing-with-imbalanced-data-in-automl/)
- [Microsoft Tech Community: Dealing with Imbalanced Data in AutoML](https://techcommunity.microsoft.com/t5/azure-ai/dealing-with-imbalanced-data-in-automl/ba-p/1625043)
- A very interesting paper on the imbalanced classes issue: [Analysis of Imbalance Strategies Recommendation using a
Meta-Learning Approach](https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_34.pdf)
- [Imbalanced Data : How to handle Imbalanced Classification Problems](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/)
- [Consume an Azure Machine Learning model deployed as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)
- [Deep learning vs. machine learning in Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-deep-learning-vs-machine-learning)
- [A Review of Azure Automated Machine Learning (AutoML)](https://medium.com/microsoftazure/a-review-of-azure-automated-machine-learning-automl-5d2f98512406)
- [Supported data guardrails](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features#supported-data-guardrails)
- [Online Video Cutter](https://online-video-cutter.com/)
