import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Customer Churn Prediction using ANN",
    layout="wide"
)

st.title("Customer Churn Prediction using ANN")

# Load Model
model = load_model("ann_model.keras")

# Load Scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load History (Optional)

try:
    with open("history.pkl", "rb") as f:
        history = pickle.load(f)
except:
    history = None

# =============================
# USER INPUTS
# =============================

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        value=50000.0
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=2
    )

with col2:

    has_credit_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    is_active_member = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        value=50000.0
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

# =============================
# ENCODING
# =============================

geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

# =============================
# PREDICTION
# =============================

if st.button("Predict Churn"):

    sample = np.array([
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_credit_card,
        is_active_member,
        estimated_salary,
        geography_germany,
        geography_spain,
        gender_male
    ]).reshape(1, -1)

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    if probability > 0.5:
        st.error(
            "Customer is likely to leave the bank."
        )
    else:
        st.success(
            "Customer is likely to stay with the bank."
        )

# =============================
# TRAINING VISUALIZATIONS
# =============================

if history:

    st.subheader("Model Performance")

    col3, col4 = st.columns(2)

    with col3:

        fig1, ax1 = plt.subplots(figsize=(4, 3))

        ax1.plot(
            history["accuracy"],
            label="Training Accuracy"
        )

        ax1.plot(
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        ax1.set_title("Accuracy Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        plt.tight_layout()

        st.pyplot(fig1)

    with col4:

        fig2, ax2 = plt.subplots(figsize=(4, 3))

        ax2.plot(
            history["loss"],
            label="Training Loss"
        )

        ax2.plot(
            history["val_loss"],
            label="Validation Loss"
        )

        ax2.set_title("Loss Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()

        st.pyplot(fig2)

# =============================
# ANN INFORMATION
# =============================

st.subheader("Model Architecture")

st.info("""
Input Layer → 11 Features

Hidden Layer 1 → 16 Neurons (ReLU)

Hidden Layer 2 → 8 Neurons (ReLU)

Output Layer → 1 Neuron (Sigmoid)
""")