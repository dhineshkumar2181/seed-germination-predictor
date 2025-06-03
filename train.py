import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load metadata
metadata_path = "dataset/metadata.xlsx"
if not os.path.exists(metadata_path):
    raise FileNotFoundError("❌ Metadata file not found!")

df = pd.read_excel(metadata_path)

# Ensure required columns exist
required_columns = {"filename", "seed_types", "health_status", "growth_period", "rainfall", "temperature"}
if not required_columns.issubset(df.columns):
    raise ValueError("❌ Metadata file is missing required columns!")

# Load images and labels
image_dir = "dataset/images"
X, y_seed, y_health, y_growth, y_rainfall, y_temperature = [], [], [], [], [], []

for _, row in df.iterrows():
    img_path = os.path.join(image_dir, row["filename"])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        X.append(img)
        y_seed.append(row["seed_types"])
        y_health.append(row["health_status"])
        y_growth.append(row["growth_period"])
        y_rainfall.append(row["rainfall"])
        y_temperature.append(row["temperature"])
    else:
        print(f"🚨 Skipping missing file: {row['filename']}")

if not X:
    raise ValueError("❌ No images found! Check file paths and metadata.")

# Convert to numpy arrays
X = np.array(X) / 255.0  # Normalize pixel values

# Convert labels to numerical values
label_encoders = {
    "seed_types": LabelEncoder(),
    "health_status": LabelEncoder(),
    "growth_period": LabelEncoder(),
    "rainfall": LabelEncoder(),
    "temperature": LabelEncoder()
}

y_seed = label_encoders["seed_types"].fit_transform(y_seed)
y_health = label_encoders["health_status"].fit_transform(y_health)
y_growth = label_encoders["growth_period"].fit_transform(y_growth)
y_rainfall = label_encoders["rainfall"].fit_transform(y_rainfall)
y_temperature = label_encoders["temperature"].fit_transform(y_temperature)

# Convert to numpy arrays
y_seed = np.array(y_seed)
y_health = np.array(y_health)
y_growth = np.array(y_growth)
y_rainfall = np.array(y_rainfall)
y_temperature = np.array(y_temperature)

# Split dataset
(X_train, X_test,
 y_seed_train, y_seed_test,
 y_health_train, y_health_test,
 y_growth_train, y_growth_test,
 y_rainfall_train, y_rainfall_test,
 y_temperature_train, y_temperature_test) = train_test_split(
    X, y_seed, y_health, y_growth, y_rainfall, y_temperature, 
    test_size=0.2, random_state=42)

# Define CNN model (currently for seed classification only)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoders["seed_types"].classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model (seed type classification only)
model.fit(X_train, y_seed_train, epochs=10, validation_data=(X_test, y_seed_test))

# Save model
model.save(r"seed_classifier.h5")
print("✅ Model training complete and saved as 'seed_classifier.h5'")
