from ingestion.dataingestion import DataIngestion
from preprocessing.prepro import DataTransformation
from Model.train import ModelTrainer
import pandas as pd
import numpy as np


obj=DataIngestion()
train_data,test_data,raw_data=obj.initiate_data_ingestion()

dt = DataTransformation()
df_array= dt.run_preprocessor(raw_data)



train_obj = ModelTrainer()
f1 = train_obj.train(df_array)
print(f1)

