"""
Hybrid LSTM with Selectable Transforms
--------------------------------------
User can combine any 2 transforms among:
- HHT (adaptive nonlinear oscillations)
- Wavelet (multi-scale transient detection)
- STFT (stable periodic and harmonic content)

Example combos:
    ["HHT", "Wavelet"]
    ["HHT", "STFT"]
    ["Wavelet", "STFT"]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------
# CONFIG
# -------------------------
WINDOW_SIZE = 12
EPOCHS = 100
BATCH_SIZE = 8

# Choose any 2 transforms
TRANSFORM_COMBO = ["HHT", "Wavelet"]   # change to ["HHT", "STFT"], ["HHT", "Wavelet"] or ["Wavelet", "STFT"]

# -------------------------
# IMPORT TRANSFORMS
# -------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transforms.hht import HHTTransform
from transforms.wavelet import WaveletTransform
from transforms.stft import STFTTransform

# Initialize based on user selection
transforms_map = {
    "HHT": HHTTransform(),
    "Wavelet": WaveletTransform(),
    "STFT": STFTTransform()
}
selected_transforms = [transforms_map[name] for name in TRANSFORM_COMBO]

# -------------------------
# LOAD SALES DATA
# -------------------------
try:
    df = pd.read_csv("abit_transforms/complex_cookie_sales.csv")
    sales = df['wavelet_pattern'].values
except FileNotFoundError:
    print("⚠️ data/sales.csv not found. Using synthetic data instead.")
    np.random.seed(42)
    time = np.arange(72)
    sales = (
        200
        + 20 * np.sin(2 * np.pi * time / 12)
        + 10 * np.sin(2 * np.pi * time / 6)
        + np.random.normal(0, 5, 72)
    )

# Normalize sales
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()

# -------------------------
# FEATURE EXTRACTION HELPERS
# -------------------------
def extract_features(transform_obj, window, max_patterns=3):
    """Extract frequency, amplitude, confidence from transform patterns."""
    patterns, _ = transform_obj.analyze(window)
    feats = []
    for p in patterns[:max_patterns]:
        feats.extend([p.frequency, p.amplitude, p.confidence])
    while len(feats) < max_patterns * 3:
        feats.append(0.0)
    return np.array(feats)

# -------------------------
# BUILD DATASET
# -------------------------
X_features, y_target = [], []

for i in range(len(sales_scaled) - WINDOW_SIZE):
    window = sales_scaled[i:i + WINDOW_SIZE]

    # Always include raw window as base
    feature_parts = [window]

    # Add each selected transform’s features
    for transform in selected_transforms:
        feats = extract_features(transform, window)
        feature_parts.append(feats)

    combined_feats = np.concatenate(feature_parts)
    X_features.append(combined_feats)
    y_target.append(sales_scaled[i + WINDOW_SIZE])

X_features = np.array(X_features)
y_target = np.array(y_target)

# Normalize features
feature_scaler = MinMaxScaler()
X_features = feature_scaler.fit_transform(X_features)

# Reshape for LSTM input
X_features = np.expand_dims(X_features, axis=1)
print(f"✅ Input shape for LSTM: {X_features.shape}")

# -------------------------
# TRAIN/TEST SPLIT
# -------------------------
split = int(0.8 * len(X_features))
X_train, X_test = X_features[:split], X_features[split:]
y_train, y_test = y_target[:split], y_target[split:]

# -------------------------
# BUILD LSTM MODEL
# -------------------------
model = Sequential([
    LSTM(96, return_sequences=True, input_shape=(X_features.shape[1], X_features.shape[2])),
    Dropout(0.15),
    LSTM(48, activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# -------------------------
# TRAIN
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

# -------------------------
# EVALUATE
# -------------------------
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Core metrics ---
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mean_sales = np.mean(y_test_inv)
mae_pct = (mae / mean_sales) * 100
rmse_pct = (rmse / mean_sales) * 100
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-8))) * 100
r2 = r2_score(y_test_inv, y_pred_inv)
corr = np.corrcoef(y_test_inv.flatten(), y_pred_inv.flatten())[0, 1]

print(f"\n📊 Hybrid LSTM ({' + '.join(TRANSFORM_COMBO)}) Results")
print("-------------------------------------------------------")
print(f"MAE:   {mae:.2f} ({mae_pct:.2f}%)")
print(f"RMSE:  {rmse:.2f} ({rmse_pct:.2f}%)")
print(f"MAPE:  {mape:.2f}%")
print(f"R²:    {r2:.4f}")
print(f"Corr:  {corr:.4f}")

# -------------------------
# PLOT RESULTS
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(y_test_inv, label='True Sales', marker='o')
plt.plot(y_pred_inv, label='Predicted Sales', marker='x')
plt.title(f"Hybrid LSTM (Raw + {' + '.join(TRANSFORM_COMBO)})")
plt.xlabel("Time Step")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
