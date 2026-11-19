import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

# ==========================================
# 1. DATA INGESTION & SIMULATION RE-EXECUTION
# ==========================================
print("Loading and preparing dataset for model training...")
excel_path = "API_Product_Information.xlsx"
df = pd.read_excel(excel_path)

def parse_bands_robust(band_str):
    if pd.isna(band_str):
        return []
    bands = []
    for b in str(band_str).split(","):
        b_clean = b.strip()
        if b_clean:
            try:
                bands.append(float(b_clean))
            except ValueError:
                pass
    return bands

df['Parsed_Bands'] = df['Approx. Raman bands (deviation of 2 cm-1 per external sources)'].apply(parse_bands_robust)

wavenumbers = np.arange(200, 3001, 1)

def simulate_spectrum(bands, wavenumber_grid, sigma=8.0, intensity_scale=1.0):
    spectrum = np.zeros_like(wavenumber_grid, dtype=float)
    for b in bands:
        peak = intensity_scale * np.exp(-((wavenumber_grid - b) ** 2) / (2 * (sigma ** 2)))
        spectrum += peak
    return spectrum

X_data, y_labels = [], []
np.random.seed(42)
for idx, row in df.iterrows():
    base_bands = row['Parsed_Bands']
    for aug_idx in range(50):
        class_type = aug_idx % 3
        if class_type == 0:
            scale = np.random.uniform(0.95, 1.05)
            bands_modified = base_bands
            label = "Authentic"
        elif class_type == 1:
            scale = np.random.uniform(0.40, 0.70)
            bands_modified = base_bands
            label = "Diluted"
        else:
            scale = np.random.uniform(0.85, 1.15)
            bands_modified = base_bands + [np.random.choice([450, 1100, 2150])]
            label = "Contaminated"
            
        spectrum = simulate_spectrum(bands_modified, wavenumbers, sigma=9.0, intensity_scale=scale)
        noise = np.random.normal(0, 0.02, size=wavenumbers.shape)
        X_data.append(spectrum + noise)
        y_labels.append(label)

X = np.array(X_data)
y = np.array(y_labels)

# Label Encoding: Authentic=0, Diluted=1, Contaminated=2
label_map = {"Authentic": 0, "Diluted": 1, "Contaminated": 2}
y_encoded = np.array([label_map[label] for label in y])

# Train-Test Split (80% Train, 20% Test with Stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# ==========================================
# 2. BASELINE MACHINE LEARNING (Random Forest)
# ==========================================
print("\nTraining Baseline Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
print("\n--- Random Forest Performance Report ---")
print(classification_report(y_test, rf_preds, target_names=list(label_map.keys())))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))

# ==========================================
# 3. DEEP LEARNING MODEL (1D-CNN Architecture)
# ==========================================
print("\nBuilding and Training 1D-CNN Model via TensorFlow/Keras...")
# Reshape data for 1D-CNN [samples, timesteps, features]
X_train_cnn = np.expand_dims(X_train, axis=-1)
X_test_cnn = np.expand_dims(X_test, axis=-1)

cnn_model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = cnn_model.fit(
    X_train_cnn, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=1
)

# Evaluate CNN
cnn_probs = cnn_model.predict(X_test_cnn)
cnn_preds = np.argmax(cnn_probs, axis=1)

print("\n--- 1D-CNN Performance Report ---")
print(classification_report(y_test, cnn_preds, target_names=list(label_map.keys())))
print("Confusion Matrix:\n", confusion_matrix(y_test, cnn_preds))