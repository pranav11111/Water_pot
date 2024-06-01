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

from src.utils import clean_data,outlier_treatment



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def obj_to_num(self,df:pd.DataFrame):

        df['Solids']  = df['Solids'].apply(clean_data)
        df['Organic_carbon'] = df['Organic_carbon'].apply(clean_data)

        for i in df.columns:
            df[i] = pd.to_numeric(df[i],errors= 'coerce')   

        return df



    def get_preprocessor(self):

        

        try:
            columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']


            pl = Pipeline(steps = [('Imputer',SimpleImputer(strategy='mean')),
                ('Scaler',MinMaxScaler())
                ])
           

            prepocessor = ColumnTransformer([ 
                ('transform',pl,columns)
                ])

            logging.info('Created Preprocessor object')

            return prepocessor

            
        except Exception as e:
            raise CustomException(e,sys)

    def run_preprocessor(self, raw_data_path):
            
        try:
            logging.info("Entered Preprocessing")

            df = pd.read_csv(raw_data_path)


            features = df.drop('Potability',axis = 1 )
            
            
            


            target = df['Potability']
            

            features = self.obj_to_num(features)
            

            logging.info('completed obj to num')

            for i in features.columns:
                features[i] = outlier_treatment(features,i)

            
            
            logging.info('completed outlier treatment')

            preproc_obj = self.get_preprocessor()
            featarray = preproc_obj.fit_transform(features)
            

            logging.info('completed imputing and scaling')

            df_array = np.c_[featarray, np.array(target)]
            

            logging.info('completed creating data')
        
            return df_array

        except Exception as e:

            raise CustomException(e,sys) 
