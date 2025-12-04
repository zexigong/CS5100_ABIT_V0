"""
Hybrid LSTM with Raw + HHT Features
-----------------------------------
Combines raw sales windows with Hilbert–Huang Transform (HHT) features.
This gives the LSTM both fine-grained trends and adaptive oscillation context.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Import your transform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transforms.hht import HHTTransform

# -------------------------
# CONFIG
# -------------------------
WINDOW_SIZE = 12     # a bit longer for richer IMFs
EPOCHS = 100
BATCH_SIZE = 8
MAX_IMFS = 3         # limit number of IMFs for consistent features

# -------------------------
# LOAD SALES DATA
# -------------------------
try:
    df = pd.read_csv("abit_transforms\complex_cookie_sales.csv")
    sales = df['complex_pattern'].values
except FileNotFoundError:
    print("⚠️ data/sales.csv not found. Using synthetic data instead.")
    np.random.seed(42)
    time = np.arange(72)
    sales = 200 + 20 * np.sin(2*np.pi*time/12) + 10 * np.sin(2*np.pi*time/6) + np.random.normal(0, 5, 72)

# Normalize
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()

# -------------------------
# HHT INITIALIZATION
# -------------------------
hht = HHTTransform()

def extract_hht_features(signal_window, max_imfs=MAX_IMFS):
    """Extract fixed-length HHT features for a given signal window."""
    patterns, artifacts = hht.analyze(signal_window)
    features = []
    for p in patterns[:max_imfs]:
        features.extend([p.frequency, p.amplitude, p.confidence])
    while len(features) < max_imfs * 3:
        features.append(0.0)
    return np.array(features)

# -------------------------
# BUILD DATASET
# -------------------------
X_features, y_target = [], []
for i in range(len(sales_scaled) - WINDOW_SIZE):
    window = sales_scaled[i:i+WINDOW_SIZE]
    hht_feats = extract_hht_features(window)

    # Combine raw window and HHT features
    combined_feats = np.concatenate([window, hht_feats])
    X_features.append(combined_feats)
    y_target.append(sales_scaled[i+WINDOW_SIZE])

X_features = np.array(X_features)
y_target = np.array(y_target)

# Normalize features again (different scales)
feature_scaler = MinMaxScaler()
X_features = feature_scaler.fit_transform(X_features)

# Reshape for LSTM input: (samples, timesteps, features)
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
    LSTM(64, activation='tanh', input_shape=(X_features.shape[1], X_features.shape[2])),
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
plt.figure(figsize=(8,5))
plt.plot(y_test_inv, label='True Sales', marker='o')
plt.plot(y_pred_inv, label='Predicted Sales', marker='x')
plt.title("Hybrid LSTM (Raw + HHT Features)")
plt.xlabel("Time Step")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
