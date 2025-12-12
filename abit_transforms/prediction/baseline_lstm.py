"""
LSTM Raw Baseline Model
-----------------------
Predicts future sales using only past raw sales data (no transforms).
This serves as a baseline to compare with ABIT-enhanced models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------
# CONFIG
# ------------------------
WINDOW_SIZE = 12   # number of past steps used for prediction
EPOCHS = 100
BATCH_SIZE = 8

# ------------------------
# LOAD SALES DATA
# ------------------------
# Replace this with sales data file

try:
    df = pd.read_csv("abit_transforms/complex_cookie_sales.csv")
    sales = df['wavelet_pattern'].values
except FileNotFoundError:
    print("⚠️ data/sales.csv not found. Generating dummy data instead.")
    np.random.seed(42)
    time = np.arange(48)
    sales = 200 + 20 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 5, 48)

# Normalize
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales.reshape(-1, 1))

# ------------------------
# CREATE SEQUENCES
# ------------------------
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(sales_scaled, WINDOW_SIZE)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------
# BUILD MODEL
# ------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.1),
    LSTM(32, activation='tanh'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print(model.summary())

# ------------------------
# TRAIN MODEL
# ------------------------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

# ------------------------
# EVALUATE MODEL
# ------------------------
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

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

print("\n📊 Baseline LSTM Results (Absolute)")
print("----------------------------------")
print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")

print("\n📈 Baseline LSTM Results (Percentage of Mean Sales)")
print("-------------------------------------------------------")
print(f"MAE (% of mean sales):  {mae_pct:.2f}%")
print(f"RMSE (% of mean sales): {rmse_pct:.2f}%")
print(f"MAPE:                   {mape:.2f}%")
print(f"R²:                     {r2:.4f}")
print(f"Corr r:                 {corr:.4f}")

# ------------------------
# PLOT RESULTS
# ------------------------
plt.figure(figsize=(8,5))
plt.plot(y_test_inv, label='True Sales', marker='o')
plt.plot(y_pred_inv, label='Predicted Sales', marker='x')
plt.title("LSTM Raw Sales Prediction")
plt.xlabel("Time Step")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
