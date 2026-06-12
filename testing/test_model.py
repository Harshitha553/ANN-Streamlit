from tensorflow.keras.models import load_model

model = load_model("ann_model.keras")

print(model.summary())