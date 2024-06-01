import re
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from src.exception import CustomException
import os
import sys


def clean_data(text):
    text = re.sub(r"[' ',()-->?¥]","",text)
    return text

def outlier_treatment(data,series):
    q1 = np.percentile(data[series], 25)
    q3 = np.percentile(data[series], 75)
    iqr = q3 - q1
    
    for idx in range(len(data[series])):
        elem = data.at[idx, series]
        if elem > q3 + iqr * 1.5:
            data.at[idx, series] = q3 + iqr * 1.5
        elif elem < q1 - iqr * 1.5:
            data.at[idx, series] = q1 - iqr * 1.5
    
    return data[series]


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def eval(model,x,y):
    pred = model.predict(x)
    f1 = f1_score(y,pred)
    return f1


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)