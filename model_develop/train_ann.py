import pandas as pd
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/Churn_Modelling.csv")

X = df.iloc[:, 3:13]

X = pd.get_dummies(
    X,
    drop_first=True
)

y = df["Exited"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(Dense(
    units=16,
    activation='relu',
    input_dim=X_train.shape[1]
))

model.add(Dense(
    units=8,
    activation='relu'
))

model.add(Dense(
    units=1,
    activation='sigmoid'
))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

model.save("ann_model.keras")

print("ANN Model Saved Successfully")
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
