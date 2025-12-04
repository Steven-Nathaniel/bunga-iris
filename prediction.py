import joblib
import pandas as pd

def predict(data):
    clf = joblib.load("rf_model.sav")
    
    df = pd.DataFrame(data, columns=[
        'SepalLengthCm', 
        'SepalWidthCm', 
        'PetalLengthCm', 
        'PetalWidthCm'
    ])
    
    return clf.predict(df)
