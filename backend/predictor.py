import pickle
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("ann_model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_churn(data):

    data = np.array(data).reshape(1,-1)

    data = scaler.transform(data)

    prediction = model.predict(data)

    return prediction[0][0]