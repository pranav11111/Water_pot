import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
import os

from preprocessing.utils import clean_data,outlier_treatment,save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_preprocessor(self):


        try:
            pipeline = Pipeline(steps=[
                ('clean_data',clean_data()),
                ('to_numeric',pd.to_numeric(errors= 'coerce'))
                ('Treat Outliers',outlier_treatment()),
                ('Impute',SimpleImputer(strategy='median'))
                ('Scaling',MinMaxScaler())

            ])
        except:
            pass