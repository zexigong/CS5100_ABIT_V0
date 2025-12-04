"""
Layered ABIT LSTM
------------------
Combines HHT, Wavelet, and STFT feature transformations
for advanced time-frequency-aware sales forecasting.

Each transform contributes a layer of interpretability:
- HHT: Adaptive nonlinear oscillations
- Wavelet: Multi-scale transient detection
- STFT: Stable periodic and harmonic content
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Import transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transforms.hht import HHTTransform
from transforms.stft import STFTTransform
from transforms.wavelet import WaveletTransform

# -------------------------
# CONFIG
# -------------------------
WINDOW_SIZE = 12
EPOCHS = 120
BATCH_SIZE = 8
MAX_IMFS = 3
MAX_WAVELETS = 3
MAX_STFT_PEAKS = 3

# -------------------------
# LOAD SALES DATA
# -------------------------
try:
    df = pd.read_csv("cookie_sales.csv")
    sales = df['chocochip'].values
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

# Normalize
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()

# -------------------------
# INITIALIZE TRANSFORMS
# -------------------------
hht = HHTTransform()
wavelet = WaveletTransform()
stft = STFTTransform()

# -------------------------
# FEATURE EXTRACTORS
# -------------------------
def extract_hht_features(window):
    patterns, _ = hht.analyze(window)
    feats = []
    for p in patterns[:MAX_IMFS]:
        feats.extend([p.frequency, p.amplitude, p.confidence])
    while len(feats) < MAX_IMFS * 3:
        feats.append(0.0)
    return feats


def extract_wavelet_features(window):
    patterns, _ = wavelet.analyze(window)
    feats = []
    for p in patterns[:MAX_WAVELETS]:
        feats.extend([p.frequency, p.amplitude, p.confidence])
    while len(feats) < MAX_WAVELETS * 3:
        feats.append(0.0)
    return feats


def extract_stft_features(window):
    patterns, _ = stft.analyze(window)
    feats = []
    for p in patterns[:MAX_STFT_PEAKS]:
        feats.extend([p.frequency, p.amplitude, p.confidence])
    while len(feats) < MAX_STFT_PEAKS * 3:
        feats.append(0.0)
    return feats


# -------------------------
# BUILD DATASET
# -------------------------
X_features, y_target = [], []
for i in range(len(sales_scaled) - WINDOW_SIZE):
    window = sales_scaled[i:i+WINDOW_SIZE]
    hht_feats = extract_hht_features(window)
    wavelet_feats = extract_wavelet_features(window)
    stft_feats = extract_stft_features(window)

    # Combine all features (raw + transforms)
    combined = np.concatenate([window, hht_feats, wavelet_feats, stft_feats])
    X_features.append(combined)
    y_target.append(sales_scaled[i + WINDOW_SIZE])

X_features = np.array(X_features)
y_target = np.array(y_target)

# Rescale features to 0-1 range
feature_scaler = MinMaxScaler()
X_features = feature_scaler.fit_transform(X_features)

# Reshape for LSTM
X_features = np.expand_dims(X_features, axis=1)
print(f"✅ Input shape for LSTM: {X_features.shape}")

# -------------------------
# TRAIN/TEST SPLIT
# -------------------------
split = int(0.8 * len(X_features))
X_train, X_test = X_features[:split], X_features[split:]
y_train, y_test = y_target[:split], y_target[split:]

# -------------------------
# BUILD LAYERED LSTM MODEL
# -------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_features.shape[1], X_features.shape[2])),
    Dropout(0.2),
    LSTM(64, activation='tanh'),
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

# --- Convert to percentage relative to mean actual value ---
mean_sales = np.mean(y_test_inv)
mae_pct = (mae / mean_sales) * 100
rmse_pct = (rmse / mean_sales) * 100

# --- Additional metrics ---
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-8))) * 100
r2 = r2_score(y_test_inv, y_pred_inv)
corr = np.corrcoef(y_test_inv.flatten(), y_pred_inv.flatten())[0, 1]

print("\n📊 ABIT Layered LSTM Results (Absolute)")
print("----------------------------------")
print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")

print("\n📈 ABIT Layered LSTM Results (Percentage of Mean Sales)")
print("-------------------------------------------------------")
print(f"MAE (% of mean sales):  {mae_pct:.2f}%")
print(f"RMSE (% of mean sales): {rmse_pct:.2f}%")
print(f"MAPE:                   {mape:.2f}%")
print(f"R²:                     {r2:.4f}")
print(f"Corr r:                 {corr:.4f}")

# -------------------------
# PLOT RESULTS
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(y_test_inv, label='True Sales', marker='o')
plt.plot(y_pred_inv, label='Predicted Sales', marker='x')
plt.title("ABIT Layered LSTM (HHT + Wavelet + STFT + Raw)")
plt.xlabel("Time Step")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
