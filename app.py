import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

# ==========================================
# PAGE CONFIG & DARK THEME STYLING
# ==========================================
st.set_page_config(
    page_title="PharmaGuard — Medicine & Vial Spectral Lab",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b111e;
            color: #e6edf3;
        }
        .header-card {
            background-color: #111927;
            border: 1px solid #1f2d42;
            border-radius: 12px;
            padding: 18px 24px;
            margin-bottom: 18px;
        }
        .section-card {
            background-color: #111927;
            border: 1px solid #1f2d42;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 16px;
        }
        .metric-box {
            background-color: #0e1624;
            border: 1px solid #22324a;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }
        .metric-val {
            font-size: 1.45rem;
            font-weight: 700;
            color: #38bdf8;
        }
        .metric-lbl {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-top: 2px;
        }
        .verdict-right {
            background-color: rgba(34, 197, 94, 0.14);
            border: 1px solid #22c55e;
            color: #86efac;
            padding: 14px 18px;
            border-radius: 10px;
            font-weight: 600;
            margin-top: 10px;
        }
        .verdict-wrong {
            background-color: rgba(239, 68, 68, 0.14);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 14px 18px;
            border-radius: 10px;
            font-weight: 600;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

WAVENUMBERS = np.arange(200, 3001, 1)
BAND_COL = "Approx. Raman bands (deviation of 2 cm-1 per external sources)"


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


def parse_custom_bands_input(text_val):
    bands = []
    for token in str(text_val).split(","):
        t = token.strip()
        if t:
            val = float(t)
            if 200.0 <= val <= 3000.0:
                bands.append(val)
    return bands


def simulate_spectrum(bands, wavenumber_grid, sigma=8.0, intensity_scale=1.0):
    spectrum = np.zeros_like(wavenumber_grid, dtype=float)
    for b in bands:
        peak = intensity_scale * np.exp(-((wavenumber_grid - b) ** 2) / (2 * (sigma ** 2)))
        spectrum += peak
    return spectrum


def apply_vial_effects(spectrum, thickness_mm, noise_std=0.02):
    """
    Simulates vial wall thickness impact on Raman signal:
    - Thicker walls slightly attenuate transmitted/collected signal intensity.
    - Adds subtle glass scattering baseline + sensor noise.
    """
    attenuation = np.exp(-0.08 * max(0.0, thickness_mm - 1.0))
    glass_baseline = 0.015 * max(0.0, thickness_mm - 0.5) * np.exp(
        -((WAVENUMBERS - 800.0) ** 2) / (2 * (250.0 ** 2))
    )
    effective_noise = noise_std * (1.0 + 0.15 * max(0.0, thickness_mm - 1.0))
    noise = np.random.normal(0.0, effective_noise, size=WAVENUMBERS.shape)
    return (spectrum * attenuation) + glass_baseline + noise


@st.cache_data
def load_compounds(excel_path="API_Product_Information.xlsx"):
    df = pd.read_excel(excel_path)
    df["Parsed_Bands"] = df[BAND_COL].apply(parse_bands_robust)
    return df


@st.cache_resource
def train_models(df, augmentations_per_sample=50, cnn_epochs=10):
    np.random.seed(42)
    tf.random.set_seed(42)

    X_data, y_labels = [], []
    for _, row in df.iterrows():
        base_bands = row["Parsed_Bands"]
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
                bands_modified = base_bands + [np.random.choice([450, 1100, 2150])]
                label = "Contaminated"

            spectrum = simulate_spectrum(
                bands_modified, WAVENUMBERS, sigma=9.0, intensity_scale=scale
            )
            noise = np.random.normal(0, 0.02, size=WAVENUMBERS.shape)
            X_data.append(spectrum + noise)
            y_labels.append(label)

    X = np.array(X_data)
    y = np.array(y_labels)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # 1D-CNN
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    cnn_model = Sequential(
        [
            Input(shape=(X_train.shape[1], 1)),
            Conv1D(filters=32, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(len(le.classes_), activation="softmax"),
        ]
    )
    cnn_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cnn_model.fit(
        X_train_cnn,
        y_train,
        epochs=cnn_epochs,
        batch_size=32,
        validation_data=(X_test_cnn, y_test),
        verbose=0,
    )
    cnn_probs = cnn_model.predict(X_test_cnn, verbose=0)
    cnn_preds = np.argmax(cnn_probs, axis=1)

    metrics = {}
    for name, preds in [("1D-CNN", cnn_preds), ("Random Forest", rf_preds)]:
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="macro", zero_division=0),
            "recall": recall_score(y_test, preds, average="macro", zero_division=0),
            "f1": f1_score(y_test, preds, average="macro", zero_division=0),
            "cm": confusion_matrix(y_test, preds),
        }

    return le, rf_model, cnn_model, metrics


if "test_history" not in st.session_state:
    st.session_state.test_history = []

# ==========================================
# HEADER
# ==========================================
st.markdown(
    """
    <div class="header-card">
        <h2 style="margin:0;">🧪 PharmaGuard — Medicine & Vial Spectral Verification Dashboard</h2>
        <p style="margin:4px 0 0 0; color:#94a3b8;">
            Manual Chemical Composition &amp; Vial Thickness Simulator | 1D-CNN &amp; Random Forest Prediction Checker
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df_compounds = load_compounds("API_Product_Information.xlsx")
except Exception as e:
    st.error(f"Could not load API_Product_Information.xlsx: {e}")
    st.stop()

with st.spinner("Training / loading 1D-CNN and Random Forest models..."):
    le, rf_model, cnn_model, all_metrics = train_models(df_compounds)

# ==========================================
# MAIN TWO-COLUMN LAYOUT
# ==========================================
left_col, right_col = st.columns([1.05, 1.25], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧾 Medicine & Vial Input Form")

    compound_list = df_compounds["Compound"].dropna().tolist()
    selected_compound = st.selectbox("Reference API Compound", compound_list)
    row_data = df_compounds[df_compounds["Compound"] == selected_compound].iloc[0]
    default_bands_str = ", ".join(str(int(b)) if float(b).is_integer() else str(b) for b in row_data["Parsed_Bands"])

    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox("Classifier Model", ["1D-CNN", "Random Forest"])
    with c2:
        expected_label = st.selectbox(
            "Expected Ground Truth",
            ["Authentic", "Diluted", "Contaminated"],
        )

    st.markdown("**🔬 Chemical / Raman Band Configuration**")
    active_bands_input = st.text_input(
        "Active Chemical Raman Bands (cm⁻¹)",
        value=default_bands_str,
        help="Edit or replace peak wavenumbers (200–3000 cm⁻¹) to test altered chemical composition.",
    )

    c3, c4 = st.columns(2)
    with c3:
        concentration_scale = st.slider(
            "Active Ingredient Intensity / Assay Scale",
            min_value=0.10,
            max_value=1.30,
            value=1.00,
            step=0.02,
            help="~1.00 = Authentic assay, ~0.40–0.70 = Diluted",
        )
    with c4:
        vial_thickness_mm = st.slider(
            "Vial Wall Thickness (mm)",
            min_value=0.5,
            max_value=4.0,
            value=1.0,
            step=0.1,
            help="Increases signal attenuation and glass scattering noise.",
        )

    extra_bands_input = st.text_input(
        "Added Contaminant / Impurity Peaks (cm⁻¹, optional)",
        value="",
        help="Example: 450, 1100, 2150 to simulate foreign chemical peaks.",
    )

    run_check = st.button("🔍 Run Prediction & Verify", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    active_m = all_metrics[model_choice]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"📈 Model Evaluation Metrics ({model_choice})")
    m1, m2, m3, m4 = st.columns(4)
    for col_box, label, val in [
        (m1, "Accuracy", f"{active_m['accuracy'] * 100:.2f}%"),
        (m2, "Precision", f"{active_m['precision'] * 100:.2f}%"),
        (m3, "Recall", f"{active_m['recall'] * 100:.2f}%"),
        (m4, "F1 Score", f"{active_m['f1'] * 100:.2f}%"),
    ]:
        col_box.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Parse user-modified bands
    try:
        modified_bands = parse_custom_bands_input(active_bands_input)
        extra_bands = parse_custom_bands_input(extra_bands_input)
    except ValueError:
        st.error("Please enter valid numeric wavenumbers separated by commas.")
        st.stop()

    combined_bands = modified_bands + extra_bands
    raw_sim = simulate_spectrum(
        combined_bands, WAVENUMBERS, sigma=9.0, intensity_scale=concentration_scale
    )
    test_spectrum = apply_vial_effects(raw_sim, vial_thickness_mm, noise_std=0.02)

    ref_spectrum = simulate_spectrum(
        row_data["Parsed_Bands"], WAVENUMBERS, sigma=9.0, intensity_scale=1.0
    )

    # Predict using selected model
    if model_choice == "1D-CNN":
        probs = cnn_model.predict(test_spectrum.reshape(1, -1, 1), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
    else:
        probs = rf_model.predict_proba(test_spectrum.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probs))

    predicted_label = le.classes_[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    is_right = predicted_label == expected_label

    if run_check:
        st.session_state.test_history.insert(
            0,
            {
                "Compound": selected_compound,
                "Model": model_choice,
                "Vial (mm)": vial_thickness_mm,
                "Scale": concentration_scale,
                "Extra Peaks": extra_bands_input if extra_bands_input.strip() else "None",
                "Expected": expected_label,
                "Predicted": predicted_label,
                "Confidence (%)": round(confidence, 2),
                "Result": "RIGHT ✅" if is_right else "WRONG ❌",
            },
        )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction Check: Right vs. Wrong")

    if is_right:
        st.markdown(
            f"""
            <div class="verdict-right">
                ✅ PREDICTED RIGHT — Model predicted <u>{predicted_label}</u> ({confidence:.2f}% confidence), matching expected <u>{expected_label}</u> at {vial_thickness_mm:.1f} mm vial thickness.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-wrong">
                ❌ PREDICTED WRONG — Model predicted <u>{predicted_label}</u> ({confidence:.2f}% confidence), but expected <u>{expected_label}</u> (Vial: {vial_thickness_mm:.1f} mm, Scale: {concentration_scale:.2f}).
            </div>
            """,
            unsafe_allow_html=True,
        )

    prob_df = pd.DataFrame(
        {"Class": le.classes_, "Probability (%)": [round(float(p) * 100, 2) for p in probs]}
    )
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# BOTTOM VISUALIZATIONS & TEST LOG
# ==========================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📊 Spectral Comparison & Confusion Matrix")
v1, v2 = st.columns([1.45, 1.0], gap="large")

with v1:
    fig_spec, ax_spec = plt.subplots(figsize=(8.5, 3.8), dpi=150)
    fig_spec.patch.set_facecolor("#111927")
    ax_spec.set_facecolor("#0e1624")
    ax_spec.plot(
        WAVENUMBERS, ref_spectrum, label="Reference Authentic API", color="#22c55e", linewidth=1.6
    )
    ax_spec.plot(
        WAVENUMBERS,
        test_spectrum,
        label=f"Modified Test ({vial_thickness_mm:.1f} mm vial)",
        color="#38bdf8",
        linestyle="--",
        linewidth=1.6,
    )
    ax_spec.set_title("Simulated Raman Spectrum", color="#e6edf3", fontsize=11)
    ax_spec.set_xlabel("Raman Shift (cm⁻¹)", color="#94a3b8")
    ax_spec.set_ylabel("Intensity (a.u.)", color="#94a3b8")
    ax_spec.tick_params(colors="#94a3b8")
    ax_spec.grid(True, linestyle=":", alpha=0.25)
    ax_spec.legend(facecolor="#111927", edgecolor="#22324a", labelcolor="#e6edf3", fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig_spec)

with v2:
    cm = active_m["cm"]
    fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.8), dpi=150)
    fig_cm.patch.set_facecolor("#111927")
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(le.classes_)))
    ax_cm.set_yticks(range(len(le.classes_)))
    ax_cm.set_xticklabels(le.classes_, color="#e6edf3", fontsize=8.5)
    ax_cm.set_yticklabels(le.classes_, color="#e6edf3", fontsize=8.5)
    ax_cm.set_xlabel("Predicted", color="#94a3b8")
    ax_cm.set_ylabel("Actual", color="#94a3b8")
    ax_cm.set_title(f"{model_choice} Confusion Matrix", color="#e6edf3", fontsize=11)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "#0b111e",
                fontweight="bold",
                fontsize=9,
            )
    plt.tight_layout()
    st.pyplot(fig_cm)

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.test_history:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📝 Logged Manual Test Runs")
    st.dataframe(pd.DataFrame(st.session_state.test_history), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)