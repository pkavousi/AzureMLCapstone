import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


import json
import numpy as np
import os
import joblib
import pandas as pd
from prep import CategoricalEncoder

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def init():
    global model
    global pipeline
    model_path =os.path.join(os.getenv('AZUREML_MODEL_DIR'),  'outputs/model.pkl')
    model = joblib.load(model_path)
    pipeline_path=os.path.join(os.getenv('AZUREML_MODEL_DIR'),'outputs/pipeline.pkl')
    pipeline = joblib.load(pipeline_path)


def decode_response(score):
    return 'Existing Customer' if int(score)==0 else 'Attrited Customer'


def predict(data):
    data = json.loads(data)['data']
    df = pd.DataFrame.from_dict(data)
    result=pipeline.transform(df)
    score = model.predict_proba(result)[:,1]
    response=model.predict(result)
    return response, np.round((1-float(score))*100,2),


def run(data):
    try:
        flag,sc = predict(data)
        return json.dumps({'result': flag[0],'Attrition Probability(%)': sc}, cls=NumpyEncoder)
    except Exception as e:
        error = str(e)
        return error
