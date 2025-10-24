# main.py (IMPROVED VERSION)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from data_preparation import load_and_preprocess_data
from demand_model import build_lstm_model
from price_optimizer import ParticleSwarmOptimizer

def create_sequences(X, y, time_steps=10):
    """
    สร้าง "หน้าต่าง" ข้อมูลแบบ Time Series
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

SEQUENCE_LENGTH = 14  # เพิ่มเป็น 14 วัน (2 สัปดาห์)

# --- 1. Data Loading and Preparation ---
DATA_FILE = r'E:\model\model\Amazon Sale Report.csv'
df = load_and_preprocess_data(DATA_FILE)

# *** UPDATED FEATURES LIST ***
features = [
    'SKU', 'Category', 'Size', 'Avg_Price', 'Max_Price', 'Min_Price', 'Std_Price',
    'day_of_week', 'month', 'week_of_year', 'day_of_month', 'quarter',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
    'is_weekend', 'is_month_start', 'is_month_end',
    'Has_Promotion', 'price_x_promotion', 'price_change_pct',
    'Qty_lag_1', 'Qty_lag_3', 'Qty_lag_7', 'Qty_lag_14',
    'Qty_roll_mean_3', 'Qty_roll_mean_7', 'Qty_roll_mean_14', 'Qty_roll_mean_30',
    'Qty_roll_std_3', 'Qty_roll_std_7', 'Qty_roll_std_14', 'Qty_roll_std_30',
    'Price_lag_1', 'Price_lag_7',
    'Promo_lag_1', 'Promo_lag_7',
    'qty_trend_7', 'qty_trend_30'
]

target = 'Total_Qty'
NUM_FEATURES = len(features)
print(f"\nTotal features: {NUM_FEATURES}")

# --- 2. Time-based Train/Val/Test Split (แก้ปัญหา Data Leakage) ---
print("\n=== Time-based Data Split ===")
df = df.sort_values(by=['SKU', 'Date'])

# หาจำนวนข้อมูลทั้งหมด
total_records = len(df)
train_end_idx = int(total_records * 0.8)   # 80% Train
val_end_idx = int(total_records * 0.9)    # 10% Val, 20% Test

df_train = df.iloc[:train_end_idx].copy()
df_val = df.iloc[train_end_idx:val_end_idx].copy()
df_test = df.iloc[val_end_idx:].copy()

print(f"Train period: {df_train['Date'].min()} to {df_train['Date'].max()}")
print(f"Val period:   {df_val['Date'].min()} to {df_val['Date'].max()}")
print(f"Test period:  {df_test['Date'].min()} to {df_test['Date'].max()}")

# --- 3. Fit Scaler เฉพาะ Train Set ---
print("\n=== Fitting Scaler on Train Set ONLY ===")
X_train_raw = df_train[features].values
y_train_raw = df_train[target].values

scaler_X = StandardScaler()
scaler_X.fit(X_train_raw)

# Scale ทุก Set
X_train_scaled = scaler_X.transform(X_train_raw)
X_val_scaled = scaler_X.transform(df_val[features].values)
X_test_scaled = scaler_X.transform(df_test[features].values)

# Log Transform Target
y_train_log = np.log1p(y_train_raw)
y_val_log = np.log1p(df_val[target].values)
y_test_raw = df_test[target].values  # เก็บ raw สำหรับ evaluation

# --- 4. สร้าง Sequences Per SKU ---
print("\n=== Creating Sequences Per SKU ===")

def create_sequences_per_sku(X_scaled, y_log, df_subset, seq_length):
    """สร้าง sequences โดยแยกตาม SKU"""
    all_X_seq = []
    all_y_seq = []
    
    for sku_code in df_subset['SKU'].unique():
        sku_indices = df_subset[df_subset['SKU'] == sku_code].index
        
        # Map indices กลับไปหา position ใน X_scaled
        sku_positions = [df_subset.index.get_loc(idx) for idx in sku_indices]
        
        if len(sku_positions) > seq_length:
            X_sku = X_scaled[sku_positions]
            y_sku = y_log[sku_positions]
            
            X_seq, y_seq = create_sequences(X_sku, y_sku, seq_length)
            
            if X_seq.shape[0] > 0:
                all_X_seq.append(X_seq)
                all_y_seq.append(y_seq)
    
    if len(all_X_seq) > 0:
        return np.concatenate(all_X_seq), np.concatenate(all_y_seq)
    else:
        return np.array([]), np.array([])

X_train_seq, y_train_seq = create_sequences_per_sku(X_train_scaled, y_train_log, df_train, SEQUENCE_LENGTH)
X_val_seq, y_val_seq = create_sequences_per_sku(X_val_scaled, y_val_log, df_val, SEQUENCE_LENGTH)
X_test_seq, y_test_seq_raw = create_sequences_per_sku(X_test_scaled, y_test_raw, df_test, SEQUENCE_LENGTH)

print(f"\nSequences created:")
print(f"X_train shape: {X_train_seq.shape}, y_train shape: {y_train_seq.shape}")
print(f"X_val shape:   {X_val_seq.shape}, y_val shape: {y_val_seq.shape}")
print(f"X_test shape:  {X_test_seq.shape}, y_test shape: {y_test_seq_raw.shape}")

# --- 5. Build and Train Model ---
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

print("\n=== Training Model ===")
history = demand_model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=64,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

demand_model.save('demand_forecasting_model_improved.h5')
print("\nModel saved!")

# --- 6. Model Evaluation ---
print("\n=== Model Evaluation ===")

# Predict
predictions_log = demand_model.predict(X_test_seq).flatten()
predictions_raw = np.expm1(predictions_log)
predictions_raw = np.maximum(0, predictions_raw)

# Metrics
mae = mean_absolute_error(y_test_seq_raw, predictions_raw)
mse = mean_squared_error(y_test_seq_raw, predictions_raw)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_seq_raw, predictions_raw)
mape = np.mean(np.abs((y_test_seq_raw - predictions_raw) / (y_test_seq_raw + 1e-6))) * 100

print(f"\nTest Set Performance:")
print(f"  MAE:  {mae:.2f} units")
print(f"  RMSE: {rmse:.2f} units")
print(f"  R²:   {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

# Additional Analysis
print(f"\nActual Demand Stats:")
print(f"  Mean: {y_test_seq_raw.mean():.2f}")
print(f"  Std:  {y_test_seq_raw.std():.2f}")
print(f"  Min:  {y_test_seq_raw.min():.2f}")
print(f"  Max:  {y_test_seq_raw.max():.2f}")

print(f"\nPredicted Demand Stats:")
print(f"  Mean: {predictions_raw.mean():.2f}")
print(f"  Std:  {predictions_raw.std():.2f}")
print(f"  Min:  {predictions_raw.min():.2f}")
print(f"  Max:  {predictions_raw.max():.2f}")

# --- 7. Visualization ---
plt.figure(figsize=(15, 5))

# Plot 1: Training History
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)

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
plt.savefig('model_evaluation.png', dpi=150)
print("\nVisualization saved to: model_evaluation.png")

# --- 8. Price Optimization ---
print("\n=== Price Optimization ===")

# เลือก Top 3 SKUs ที่มีข้อมูลมากที่สุด
sku_counts = df['SKU'].value_counts()
target_skus_encoded = sku_counts.head(3).index.values
PRODUCT_COSTS = np.array([300, 400, 600])
NUM_PRODUCTS = len(target_skus_encoded)

print(f"Optimizing prices for SKUs: {target_skus_encoded}")

# สร้าง Base Sequence จากข้อมูล Test Set ล่าสุด
base_features = X_test_scaled[-SEQUENCE_LENGTH+1:].copy()

def profit_objective_function(prices):
    """
    ฟังก์ชันคำนวณกำไรรวม (ที่จะ Maximize)
    """
    model_inputs = []
    
    for i, price in enumerate(prices):
        # สร้าง feature vector สำหรับวันพรุ่งนี้
        tomorrow_features = X_test_scaled[-1].copy()
        
        # อัปเดตค่าที่เกี่ยวกับ SKU และ Price
        tomorrow_features[0] = target_skus_encoded[i]  # SKU
        tomorrow_features[3] = (price - scaler_X.mean_[3]) / scaler_X.scale_[3]  # Scaled Price
        
        # สร้าง Sequence (เอา base + tomorrow)
        sequence = np.vstack([base_features, tomorrow_features])
        model_inputs.append(sequence)
    
    model_inputs_array = np.array(model_inputs)
    
    # Predict Demand
    predictions_log = demand_model.predict(model_inputs_array, verbose=0).flatten()
    predictions_raw = np.expm1(predictions_log)
    predictions_raw = np.maximum(0, predictions_raw)
    
    # คำนวณกำไร
    total_profit = np.sum((prices - PRODUCT_COSTS) * predictions_raw)
    
    return -total_profit  # ติดลบเพราะ PSO minimize

# Price Bounds
price_bounds = [
    (400, 800),
    (500, 1000),
    (700, 1500)
]

optimizer = ParticleSwarmOptimizer(
    objective_function=profit_objective_function,
    bounds=price_bounds,
    num_particles=50,
    max_iter=100
)

optimal_prices, max_profit = optimizer.optimize()

print(f"\n{'='*50}")
print(f"OPTIMIZATION RESULTS:")
print(f"{'='*50}")
print(f"Optimal Prices: {np.round(optimal_prices, 2)}")
print(f"Maximum Profit: ฿{-max_profit:,.2f}")
print(f"{'='*50}")

# ทดสอบ Demand ที่ราคาที่หาได้
test_demands_log = demand_model.predict(
    np.array([
        np.vstack([base_features, 
                   [(target_skus_encoded[i] - scaler_X.mean_[0]) / scaler_X.scale_[0],
                    0, 0,  # Category, Size (placeholder)
                    (optimal_prices[i] - scaler_X.mean_[3]) / scaler_X.scale_[3]] + 
                   [0] * (NUM_FEATURES - 4)])  # Fill remaining features
        for i in range(NUM_PRODUCTS)
    ]),
    verbose=0
).flatten()

test_demands = np.expm1(test_demands_log)
test_demands = np.maximum(0, test_demands)

print(f"\nPredicted Demands at Optimal Prices:")
for i in range(NUM_PRODUCTS):
    print(f"  SKU {target_skus_encoded[i]}: {test_demands[i]:.2f} units @ ฿{optimal_prices[i]:.2f}")
    print(f"    -> Profit: ฿{(optimal_prices[i] - PRODUCT_COSTS[i]) * test_demands[i]:,.2f}")
