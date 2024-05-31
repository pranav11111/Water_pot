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

from preprocessing.utils import clean_data,outlier_treatment



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

           

            prepocessor = ColumnTransformer([ 
                ('Impute',SimpleImputer(strategy='median'),columns),
                ('Scaling',MinMaxScaler(),columns)
                ])

            logging.info('Created Preprocessor object')

            return prepocessor

            
        except Exception as e:
            raise CustomException(e,sys)

    def run_preprocessor(self, train_path,test_path):
            
        try:
            logging.info("Entered Preprocessing")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            features_train = train_df.drop('Potability',axis = 1 )
            features_test = test_df.drop('Potability',axis =1)
            

            target_train = train_df['Potability']
            target_test = test_df['Potability']

            features_train = self.obj_to_num(features_train)
            features_test  = self.obj_to_num(features_test)

            logging.info('completed obj to num')

            for i in features_train.columns:
                features_train[i] = outlier_treatment(features_train,i)

            for i in features_test.columns:
                features_test[i] = outlier_treatment(features_test,i)
            
            logging.info('completed outlier treatment')

            preproc_obj = self.get_preprocessor()
            feat_train_array = preproc_obj.fit_transform(features_train)
            feat_test_array = preproc_obj.fit_transform(features_test)

            logging.info('completed imputing and scaling')

            train_array = np.c_[feat_train_array,np.array(target_train)]
            test_array = np.c_[feat_test_array,np.array(target_test)]

            logging.info('completed creating data')
        
            return train_array,test_array

        except Exception as e:

            raise CustomException(e,sys) 
