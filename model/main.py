# main.py (WEEKLY VERSION)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- !! 1. USE WEEKLY AGGREGATOR !! ---
from aggregate_weekly import aggregate_data_weekly
from demand_model import build_lstm_model
# (Price Optimizer ถูกปิดใช้งานชั่วคราว)
# from price_optimizer import ParticleSwarmOptimizer

def create_sequences(X, y, time_steps=10):
    """
    สร้าง "หน้าต่าง" ข้อมูลแบบ Time Series (ฟังก์ชันนี้ใช้ได้เหมือนเดิม)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- !! 2. ADJUST SEQUENCE LENGTH !! ---
# 14 (วัน) ไม่เหมาะกับรายสัปดาห์
# เราจะใช้ 4 (สัปดาห์) เพื่อมองย้อนหลังประมาณ 1 เดือน
SEQUENCE_LENGTH = 4

# --- !! 3. LOAD WEEKLY DATA !! ---
DATA_FILE = r'E:\model\model\Amazon Sale Report.csv' # (ตรวจสอบ Path นี้)
print("\n=== Loading and Aggregating Weekly Data ===")
# df = load_and_preprocess_data(DATA_FILE) # <-- OLD (Daily)
df = aggregate_data_weekly(DATA_FILE) # <-- NEW (Weekly)

# --- !! 4. NEW WEEKLY FEATURE ENGINEERING !! ---
# (เราต้องสร้าง Feature ใหม่ที่เหมาะกับรายสัปดาห์)
print("\n=== Creating WEEKLY Features ===")
df = df.sort_values(by=['SKU', 'Date'])

# Time Features (Weekly)
# (Date อยู่ใน index, ต้อง reset ก่อน)
df = df.reset_index()
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(float)
df['month'] = df['Date'].dt.month.astype(float)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# Lag Features (Weekly) - ใช้ .shift(1) เพื่อป้องกัน Data Leakage
df['Qty_lag_1'] = df.groupby('SKU')['Total_Qty'].shift(1) # 1 สัปดาห์ก่อน
df['Qty_lag_2'] = df.groupby('SKU')['Total_Qty'].shift(2) # 2 สัปดาห์ก่อน
df['Qty_lag_4'] = df.groupby('SKU')['Total_Qty'].shift(4) # 4 สัปดาห์ก่อน

# Rolling Mean Features (Weekly)
df['Qty_roll_mean_4'] = df.groupby('SKU')['Total_Qty'].shift(1).rolling(window=4, min_periods=1).mean()
df['Qty_roll_mean_8'] = df.groupby('SKU')['Total_Qty'].shift(1).rolling(window=8, min_periods=1).mean()

# Price Features (Weekly)
df['Price_lag_1'] = df.groupby('SKU')['Avg_Price'].shift(1)
df['price_change_pct'] = (df['Avg_Price'] - df['Price_lag_1']) / (df['Price_lag_1'] + 1e-6)

# (Drop rows with NaNs ที่เกิดจากการสร้าง Lag)
print(f"Original shape: {df.shape}")
df = df.dropna()
print(f"Shape after dropping NaNs: {df.shape}")

# --- !! 5. NEW FEATURE LIST (Weekly) !! ---
features = [
    'SKU', 'Category', 'Size', 'Avg_Price', 'Max_Price', 'Min_Price',
    'week_of_year', 'month', 'month_sin', 'month_cos',
    'Qty_lag_1', 'Qty_lag_2', 'Qty_lag_4',
    'Qty_roll_mean_4', 'Qty_roll_mean_8',
    'Price_lag_1', 'price_change_pct'
]

target = 'Total_Qty'
NUM_FEATURES = len(features)
print(f"\nTotal weekly features: {NUM_FEATURES}")

# --- 6. Time-based Train/Val/Test Split ---
print("\n=== Time-based Data Split ===")
df = df.sort_values(by=['SKU', 'Date'])

total_records = len(df)
train_end_idx = int(total_records * 0.7)
val_end_idx = int(total_records * 0.85)

df_train = df.iloc[:train_end_idx].copy()
df_val = df.iloc[train_end_idx:val_end_idx].copy()
df_test = df.iloc[val_end_idx:].copy()

print(f"Train period: {df_train['Date'].min()} to {df_train['Date'].max()}")
print(f"Val period:   {df_val['Date'].min()} to {df_val['Date'].max()}")
print(f"Test period:  {df_test['Date'].min()} to {df_test['Date'].max()}")

# --- 7. Fit Scaler ---
print("\n=== Fitting Scaler on Train Set ONLY ===")
X_train_raw = df_train[features].values
y_train_raw = df_train[target].values

scaler_X = StandardScaler()
scaler_X.fit(X_train_raw)

X_train_scaled = scaler_X.transform(X_train_raw)
X_val_scaled = scaler_X.transform(df_val[features].values)
X_test_scaled = scaler_X.transform(df_test[features].values)

# (เราจะยังเทรนบน Raw Values + Tweedie Loss)
print("\nTraining on RAW target values (No Log Transform).")
y_train_target = y_train_raw
y_val_target = df_val[target].values
y_test_raw = df_test[target].values

# --- 8. สร้าง Sequences Per SKU ---
print("\n=== Creating Sequences Per SKU ===")

def create_sequences_per_sku(X_scaled, y_target, df_subset, seq_length):
    """สร้าง sequences โดยแยกตาม SKU"""
    all_X_seq = []
    all_y_seq = []
    
    # (ต้องมั่นใจว่า index ของ df_subset ตรงกับ X_scaled)
    df_subset = df_subset.reset_index(drop=True)
    
    for sku_code in df_subset['SKU'].unique():
        sku_indices = df_subset[df_subset['SKU'] == sku_code].index
        
        if len(sku_indices) > seq_length:
            X_sku = X_scaled[sku_indices]
            y_sku = y_target[sku_indices]
            
            X_seq, y_seq = create_sequences(X_sku, y_sku, seq_length)
            
            if X_seq.shape[0] > 0:
                all_X_seq.append(X_seq)
                all_y_seq.append(y_seq)
    
    if len(all_X_seq) > 0:
        return np.concatenate(all_X_seq), np.concatenate(all_y_seq)
    else:
        # (ป้องกัน Error ถ้าไม่มีข้อมูล)
        return np.empty((0, seq_length, X_scaled.shape[1])), np.empty((0,))

X_train_seq, y_train_seq = create_sequences_per_sku(X_train_scaled, y_train_target, df_train, SEQUENCE_LENGTH)
X_test_seq, y_test_seq_raw = create_sequences_per_sku(X_test_scaled, y_test_raw, df_test, SEQUENCE_LENGTH)

# (จัดการ Validation Set แยกกัน)
X_val_scaled_full = scaler_X.transform(df_val[features].values)
y_val_target_full = df_val[target].values
X_val_seq, y_val_seq = create_sequences_per_sku(X_val_scaled_full, y_val_target_full, df_val, SEQUENCE_LENGTH)


print(f"\nSequences created:")
print(f"X_train shape: {X_train_seq.shape}, y_train shape: {y_train_seq.shape}")
print(f"X_val shape:   {X_val_seq.shape}, y_val shape: {y_val_seq.shape}")
print(f"X_test shape:  {X_test_seq.shape}, y_test shape: {y_test_seq_raw.shape}")

# (ป้องกันการเทรนถ้าไม่มีข้อมูล)
if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
    raise ValueError("Failed to create sequences. Check data or SEQUENCE_LENGTH.")

# --- 9. Build and Train Model ---
input_shape = (SEQUENCE_LENGTH, NUM_FEATURES)
demand_model = build_lstm_model(input_shape)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print("\n=== Training Model (Weekly) ===")
history = demand_model.fit(
    X_train_seq, y_train_seq,
    epochs=150,
    batch_size=32, # (Batch size 32 น่าจะโอเค เพราะข้อมูลน้อยลง)
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

demand_model.save('demand_forecasting_model_weekly.h5')
print("\nModel saved!")

# --- 10. Model Evaluation ---
print("\n=== Model Evaluation (Weekly) ===")

predictions_raw = demand_model.predict(X_test_seq).flatten()
predictions_raw = np.maximum(0, predictions_raw)

# Metrics
mae = mean_absolute_error(y_test_seq_raw, predictions_raw)
mse = mean_squared_error(y_test_seq_raw, predictions_raw)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_seq_raw, predictions_raw)

# (แก้ไข MAPE Logic)
mask = y_test_seq_raw > 1e-6
if np.sum(mask) > 0:
    mape = np.mean(np.abs((y_test_seq_raw[mask] - predictions_raw[mask]) / y_test_seq_raw[mask])) * 100
else:
    mape = 0.0 # (ป้องกันหารด้วย 0)

print(f"\nTest Set Performance (Weekly):")
print(f"  MAE:  {mae:.2f} units")
print(f"  RMSE: {rmse:.2f} units")
print(f"  R²:   {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

# Additional Analysis
print(f"\nActual Demand Stats (Weekly):")
print(f"  Mean: {y_test_seq_raw.mean():.2f}")
print(f"  Std:  {y_test_seq_raw.std():.2f}")
print(f"  Min:  {y_test_seq_raw.min():.2f}")
print(f"  Max:  {y_test_seq_raw.max():.2f}")

print(f"\nPredicted Demand Stats (Weekly):")
print(f"  Mean: {predictions_raw.mean():.2f}")
print(f"  Std:  {predictions_raw.std():.2f}")
print(f"  Min:  {predictions_raw.min():.2f}")
print(f"  Max:  {predictions_raw.max():.2f}")

# --- 11. Visualization ---
plt.figure(figsize=(15, 5))

# Plot 1: Training History
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History (Weekly)')
plt.legend()
plt.grid(True)

# (Plot 2 และ 3 เหมือนเดิม)
# Plot 2: Actual vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(y_test_seq_raw, predictions_raw, alpha=0.5)
plt.plot([y_test_seq_raw.min(), y_test_seq_raw.max()],
         [y_test_seq_raw.min(), y_test_seq_raw.max()], 'r--', lw=2)
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title(f'Actual vs Predicted (R²={r2:.4f})')
plt.grid(True)

# Plot 3: Residuals
plt.subplot(1, 3, 3)
residuals = y_test_seq_raw - predictions_raw
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True)

plt.tight_layout()
# (แก้ Path ที่บันทึก)
plt.savefig('model_evaluation_weekly.png', dpi=150)
print("\nVisualization saved to: model_evaluation_weekly.png")

# --- 12. Price Optimization (SKIPPED) ---
print("\n=== Price Optimization (SKIPPED) ===")
print("Price Optimization logic is complex and must be re-written for WEEKLY features.")
print("Skipping this section for now...")
print("The trained model 'demand_forecasting_model_weekly.h5' is saved.")