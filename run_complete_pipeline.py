import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# ==========================================
# 1. DATA INGESTION & SPECTRAL SIMULATION
# ==========================================
print("Loading and parsing API_Product_Information.xlsx...")
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

print("Generating synthetic spectral curves and augmenting data...")
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

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 2. LABEL ENCODING
# ==========================================
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# ==========================================
# 3. BASELINE MODEL: RANDOM FOREST
# ==========================================
print("\nTraining Random Forest baseline model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_encoded)

rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test_encoded, rf_preds)
print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%\n")
print("--- Random Forest Classification Report ---")
print(classification_report(y_test_encoded, rf_preds, target_names=le.classes_))

# ==========================================
# 4. DEEP LEARNING MODEL: 1D-CNN
# ==========================================
# Reshape data for 1D-CNN input: (samples, timesteps, features)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Building and training 1D-CNN model...")
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Authentic, Diluted, Contaminated
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the CNN model
history = cnn_model.fit(
    X_train_cnn, y_train_encoded,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_cnn, y_test_encoded),
    verbose=1
)

# Evaluate CNN model
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_encoded, verbose=0)
print(f"\n1D-CNN Test Accuracy: {cnn_acc * 100:.2f}%")


# Generate detailed classification report for CNN
cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
print("\n--- 1D-CNN Classification Report ---")
print(classification_report(y_test_encoded, cnn_preds, target_names=le.classes_))

# ==========================================
# 4.1 GENERATE MODEL ACCURACY & LOSS CURVES
# ==========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_training_curves.png', dpi=300)
plt.show()
print("Training curves saved successfully as 'model_training_curves.png'!")