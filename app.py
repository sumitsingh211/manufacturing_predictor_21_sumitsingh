# ===============================
# 📦 REQUIRED LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 1️⃣ LOAD MODEL & DATA
# ===============================
# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Load dataset to get column info and defaults
df = pd.read_csv('manufacturing_dataset_1000_samples.csv', parse_dates=['Timestamp'])

# Force datetime conversion (in case pandas missed it)
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df = df.drop(columns=['Timestamp'])

# Features and target
target = 'Parts_Per_Hour'
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# ===============================
# 2️⃣ STREAMLIT UI
# ===============================
st.title("📊 Manufacturing Parts Per Hour Predictor")
st.write("Provide the feature values below to predict Parts_Per_Hour:")

# Create input widgets dynamically
input_data = {}
for col in X.columns:
    if col in numeric_features:
        default_val = float(X[col].mean())
        input_data[col] = st.number_input(f"{col}", value=default_val)
    else:  # categorical
        default_val = X[col].mode()[0]
        options = list(X[col].dropna().unique())
        try:
            default_index = options.index(default_val)
        except ValueError:
            default_index = 0
        input_data[col] = st.selectbox(f"{col}", options=options, index=default_index)

# ===============================
# 3️⃣ PREDICT BUTTON
# ===============================
if st.button("Predict Parts_Per_Hour"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.success(f"🧠 Predicted Parts_Per_Hour: {prediction:.2f}")
    st.info(f"📊 Dataset average Parts_Per_Hour: {y.mean():.2f}")
    st.warning(f"🔍 Difference from average: {abs(prediction - y.mean()):.2f}")