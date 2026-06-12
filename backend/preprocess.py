import pickle

from sklearn.preprocessing import StandardScaler

def preprocess_data(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled