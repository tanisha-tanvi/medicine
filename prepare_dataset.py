import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DATA INGESTION & PARSING
# ==========================================
print("Loading and parsing API_Product_Information.xlsx...")
excel_path = "API_Product_Information.xlsx"
df = pd.read_excel(excel_path)

def parse_bands_robust(band_str):
    """Convert comma-separated Raman band text into a list of floats, ignoring empty entries."""
    if pd.isna(band_str):
        return []
    bands = []
    for b in str(band_str).split(","):
        b_clean = b.strip()
        if b_clean: # Check if string is not empty
            try:
                bands.append(float(b_clean))
            except ValueError:
                pass
    return bands

df['Parsed_Bands'] = df['Approx. Raman bands (deviation of 2 cm-1 per external sources)'].apply(parse_bands_robust)
print(f"Successfully loaded {len(df)} compound records[cite: 1].\n")

# ==========================================
# 2. SPECTRAL CURVE SIMULATION
# ==========================================
# Define continuous wave vector grid from 200 cm-1 to 3000 cm-1 (1 cm-1 resolution)
wavenumbers = np.arange(200, 3001, 1)

def simulate_spectrum(bands, wavenumber_grid, sigma=8.0, intensity_scale=1.0):
    """Generate continuous spectral curve using Gaussian distribution around peak bands."""
    spectrum = np.zeros_like(wavenumber_grid, dtype=float)
    for b in bands:
        # Gaussian peak centered at b
        peak = intensity_scale * np.exp(-((wavenumber_grid - b) ** 2) / (2 * (sigma ** 2)))
        spectrum += peak
    return spectrum

# ==========================================
# 3. CLASS LABELING & DATA AUGMENTATION
# ==========================================
print("Generating synthetic spectral curves and augmenting data...")
X_data = []
y_labels = []
compound_names = []

# Classes: Authentic, Diluted, Contaminated
np.random.seed(42)
augmentations_per_sample = 50

for idx, row in df.iterrows():
    base_bands = row['Parsed_Bands']
    comp_name = row['Compound']
    
    for aug_idx in range(augmentations_per_sample):
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
            extra_band = np.random.choice([450, 1100, 2150])
            bands_modified = base_bands + [extra_band]
            label = "Contaminated"
            
        # Generate spectral curve and add Gaussian noise
        spectrum = simulate_spectrum(bands_modified, wavenumbers, sigma=9.0, intensity_scale=scale)
        noise = np.random.normal(0, 0.02, size=wavenumbers.shape)
        augmented_spectrum = spectrum + noise
        
        X_data.append(augmented_spectrum)
        y_labels.append(label)
        compound_names.append(comp_name)

X = np.array(X_data)
y = np.array(y_labels)

print(f"Generated total feature matrix shape: {X.shape}")
print(f"Total label array shape: {y.shape}\n")

# ==========================================
# 4. DATA SPLITTING (80% Train / 20% Test)
# ==========================================
print("Performing stratified random split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training feature set shape: {X_train.shape}")
print(f"Testing feature set shape: {X_test.shape}")
print("Dataset preparation complete!")