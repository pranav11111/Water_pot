from src.dataingestion import DataIngestion
from preprocessing.prepro import DataTransformation
import pandas as pd
import numpy as np


obj=DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()

dt = DataTransformation()
train_array, test_array = dt.run_preprocessor(train_data,test_data)

print(test_array[0])