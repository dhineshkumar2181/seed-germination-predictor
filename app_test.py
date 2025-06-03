import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import pandas as pd

# Load trained model
model = tf.keras.models.load_model("seed_classifier.h5")
model.summary()
st.write(tf.__version__)

# Define label mappings (update as needed)
seed_types = {0: "Coffee bean", 1: "Corn", 2: "Soy bean"}  # Update based on training
health_status = {0: "Healthy", 1: "Unhealthy"}

# Load metadata from Excel
metadata_file = "dataset/metadata.xlsx"

def get_metadata_info(seed_type):
    try:
        df = pd.read_excel(metadata_file)
        df.columns = df.columns.str.strip()  # Normalize column names
        row = df[df["seed_types"] == seed_type]
        if not row.empty:
            growth = row.iloc[0]["growth_period"]
            rainfall = row.iloc[0]["rainfall"]
            temperature = row.iloc[0]["temperature"]
            return growth, rainfall, temperature
        else:
            return "Unknown", "Unknown", "Unknown"
    except Exception as e:
        return f"Error: {e}", "Error", "Error"

# Streamlit UI
st.title("🌱 Seed Classification App")
st.write("Upload an image to predict seed type, health status, and view additional metadata.")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize
    img_array = np.expand_dims(img, axis=0)  # Model expects batch size

    # Get Predictions
    predictions = model.predict(img_array)

    # Extract predictions
    seed_pred = np.argmax(predictions[0])
    health_pred = int(round(predictions[0][1]))  # Adjust this if your model outputs multiple heads

    # Get metadata info
    seed_label = seed_types.get(seed_pred, "Unknown")
    growth_period, rainfall, temperature = get_metadata_info(seed_label)

    # Display Results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"### 🧬 Seed Type: {seed_label}")
    st.write(f"### 🌡️ Health Status: {health_status.get(health_pred, 'Unknown')}")
    st.write(f"### 🌱 Growth Period: {growth_period} days")
    st.write(f"### 🌧️ Rainfall Requirement: {rainfall}")
    st.write(f"### 🌞 Temperature Range: {temperature}")
