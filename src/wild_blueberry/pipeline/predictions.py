
import joblib
import numpy as np 
import pandas as pd 
import pickle
from pathlib import Path 


class PredictionPipeline:
    def __init__(self):
        with open("final_model/blueberry_model.pkl", "rb") as model_file:
            self.model = joblib.load(model_file)


    def predict(self, data):
        prediction = self.model.predict(data)


        return prediction
