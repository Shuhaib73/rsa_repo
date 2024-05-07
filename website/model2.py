# Import libraries 
import pandas as pd
import numpy as np
import re
import pickle


class Predict_Message:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, message):
        try:
            with open(self.model_path, 'rb') as f:
                model1_loaded = pickle.load(f)

            prediction = model1_loaded.predict([message])

            return prediction[0]

        except Exception as e:
            print(f"The error : {e}")        
     