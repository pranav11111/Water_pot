import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,eval
from imblearn.over_sampling import SMOTE

@dataclass
class ModelTrainerConfig:
    train_model_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()

    def train(self,df):
        
        
        try:
            
            x = df[:,:-1]
            y = df[:,-1]    

            osm = SMOTE()
            x_over,y_over = osm.fit_resample(x,y)

            x_train,x_test,y_train,y_test = train_test_split(x_over,y_over)


            logging.info('train test split done')

            model = RandomForestClassifier(class_weight = "balanced", max_depth= 250, max_features ="log2", min_impurity_decrease= 0, 
                              min_samples_split= 2)
            

            model.fit(x_train,y_train)

            logging.info('Model Created')

            save_object(file_path= self.model_config.train_model_path,obj= model)

            f1 = eval(model,x_test,y_test)
            return f1





        except Exception as e:
            raise CustomException(e,sys)
            



