import pandas as pd
import numpy as np
from src.exception import CustomException
import sys
from src.utils import load_object




class CustomData:
    def __init__(self,
                ph:float,
                Hardness:float, 
                Solids:float, 
                Chloramines:float, 
                Sulfate:float, 
                Conductivity:float,
                Organic_carbon:float,
                Trihalomethanes:float, 
                Turbidity:float
                 
                 ): 
        self.ph = ph
        self.Hardness = Hardness
        self.Solids = Solids 
        self.Chloramines = Chloramines 
        self.Sulfate = Sulfate
        self.Conductivity = Conductivity
        self.Organic_carbon = Organic_carbon
        self.Trihalomethanes = Trihalomethanes
        self.Turbidity = Turbidity

    def get_dataframe(self):
        try:
            d = {
                'ph':[self.ph],
                'Hardness':[self.Hardness],
                'Solids':[self.Solids],
                'Chloramines':[self.Chloramines],
                'Sulfate':[self.Sulfate],
                'Conductivity':[self.Conductivity],
                'Organic_carbon':[self.Organic_carbon],
                'Trihalomethanes':[self.Trihalomethanes],
                'Turbidity':[self.Turbidity]
            }

            return pd.DataFrame(d)
        except Exception as e:
            raise CustomException(e,sys)



class predicit_pipeline:
    def __init__(self):
        pass


    def predict(self,features):
        model_path = 'artifacts\model.pkl'
        preproc_path = 'artifacts\proprocessor.pkl'
        model = load_object(model_path)
        preprocessor = load_object(preproc_path)
        ready_df = preprocessor.transform(features)
        pred = model.predict(ready_df)
        return pred
        