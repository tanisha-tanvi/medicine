import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data and simulate representative curves
excel_path = "API_Product_Information.xlsx"
df = pd.read_excel(excel_path)

def parse_bands_robust(band_str):
    if pd.isna(band_str):
        return []
    return [float(b.strip()) for b in str(band_str).split(",") if b.strip()]

df['Parsed_Bands'] = df['Approx. Raman bands (deviation of 2 cm-1 per external sources)'].apply(parse_bands_robust)
wavenumbers = np.arange(200, 3001, 1)

def simulate_spectrum(bands, wavenumber_grid, sigma=8.0, intensity_scale=1.0):
    spectrum = np.zeros_like(wavenumber_grid, dtype=float)
    for b in bands:
        peak = intensity_scale * np.exp(-((wavenumber_grid - b) ** 2) / (2 * (sigma ** 2)))
        spectrum += peak
    # Normalize to max 1.0 to prevent flat-topped clipping when peaks sum up
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)
    return spectrum

# Pick the first compound (1,3-Dimethyl-2-imidazolidinone)
sample_row = df.iloc[0]
base_bands = sample_row['Parsed_Bands']
comp_name = sample_row['Compound']

# Generate Authentic, Diluted, and Contaminated signatures with clear separation
spec_authentic = simulate_spectrum(base_bands, wavenumbers, intensity_scale=1.0)
spec_diluted = simulate_spectrum(base_bands, wavenumbers, intensity_scale=0.7) # scaled height
extra_band = [450, 1100, 2150]
spec_contaminated = simulate_spectrum(base_bands + extra_band, wavenumbers, intensity_scale=1.0)

# Plotting comparative signatures
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(wavenumbers, spec_authentic, label='Authentic Compound', color='#2ca02c', linewidth=2, alpha=0.9)
plt.plot(wavenumbers, spec_diluted, label='Diluted / Substandard (Assay Reduction)', color='#ff7f0e', linestyle='--', linewidth=1.8, alpha=0.9)
plt.plot(wavenumbers, spec_contaminated, label='Contaminated (Extra Impurity Peaks)', color='#d62728', linestyle=':', linewidth=2, alpha=0.9)

plt.title(f'Normalized Raman Spectral Signatures: {comp_name}', fontsize=12, fontweight='bold', pad=12)
plt.xlabel(r'Raman Shift ($\text{cm}^{-1}$)', fontsize=11)
plt.ylabel('Normalized Intensity (a.u.)', fontsize=11)
plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlim(180, 3020)
plt.ylim(-0.05, 1.1)
plt.tight_layout()

# Save figure at 300 DPI for publication
output_filename = 'raman_spectral_comparison_publication.png'
plt.savefig(output_filename, dpi=300)
print(f"Publication-ready figure saved successfully as '{output_filename}' at 300 DPI!")
plt.show()