# main.py (ULTIMATE FIX - Windows Compatible)
import numpy as np
import pandas as pd
import os
import glob
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_preparation import load_and_preprocess_data
from demand_model import build_lstm_model
from price_optimizer import ParticleSwarmOptimizer

def find_csv_file():
    """หาไฟล์ CSV อัตโนมัติ"""
    print("🔍 Searching for CSV file...")
    
    search_patterns = [
        r"E:\model\Amazon Sale Report.csv",
        r"E:\model\model\Amazon Sale Report.csv",
        r"E:\model\data\Amazon Sale Report.csv",
        "./Amazon Sale Report.csv",
        "../Amazon Sale Report.csv",
    ]
    
    for pattern in search_patterns:
        if os.path.exists(pattern):
            print(f"✅ Found: {pattern}")
            return pattern
    
    print("   Searching recursively...")
    search_dirs = [r"E:\model", ".", ".."]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            csv_files = glob.glob(
                os.path.join(search_dir, "**", "*amazon*.csv"), 
                recursive=True
            )
            csv_files += glob.glob(
                os.path.join(search_dir, "**", "*sale*.csv"), 
                recursive=True
            )
            
            if csv_files:
                print(f"✅ Found: {csv_files[0]}")
                return csv_files[0]
    
    print("\n❌ CSV file not found automatically!")
    csv_path = input("Enter full path to CSV: ").strip().strip('"')
    
    if os.path.exists(csv_path):
        return csv_path
    else:
        raise FileNotFoundError(f"File not found: {csv_path}")

def create_sequences(X, y, time_steps=10):
    """สร้าง sequences สำหรับ LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def safe_mape(y_true, y_pred, epsilon=1.0):
    """
    คำนวณ MAPE ที่ปลอดภัย (ไม่ระเบิดเมื่อมีค่า 0)
    """
    # ใช้เฉพาะค่าที่ > epsilon
    mask = y_true > epsilon
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# =====================================
# MAIN EXECUTION
# =====================================

print("=" * 70)
print("DEMAND FORECASTING MODEL - ULTIMATE FIX VERSION")
print("=" * 70)

SEQUENCE_LENGTH = 10  # ลดลงเหลือ 10 วัน (เพราะข้อมูลน้อย)

# --- 1. Find and Load Data ---
try:
    DATA_FILE = find_csv_file()
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    exit(1)

print(f"\n📂 Using file: {DATA_FILE}")
print("=" * 70)

df = load_and_preprocess_data(DATA_FILE)

# ตรวจสอบข้อมูล
print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)
print(f"Total records: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Days: {(df['Date'].max() - df['Date'].min()).days}")
print(f"Unique SKUs: {df['SKU'].nunique()}")

print(f"\nTarget variable (Total_Qty) statistics:")
print(f"  Mean:   {df['Total_Qty'].mean():.2f}")
print(f"  Median: {df['Total_Qty'].median():.2f}")
print(f"  Std:    {df['Total_Qty'].std():.2f}")
print(f"  Min:    {df['Total_Qty'].min():.2f}")
print(f"  Max:    {df['Total_Qty'].max():.2f}")
print(f"  Zero values: {(df['Total_Qty'] == 0).sum()} ({(df['Total_Qty'] == 0).sum()/len(df)*100:.1f}%)")

# ⚠️ ปัญหาหลัก: ข้อมูลมีค่า 0 เยอะมาก!
zero_pct = (df['Total_Qty'] == 0).sum() / len(df) * 100
if zero_pct > 50:
    print(f"\n⚠️  WARNING: {zero_pct:.1f}% of data has zero demand!")
    print("   This will significantly reduce model performance.")
    print("   Consider:")
    print("   1. Removing SKUs with too many zeros")
    print("   2. Using classification (demand vs no demand) + regression")
    print("   3. Aggregating to weekly instead of daily")

# แก้ไข: ลบ SKUs ที่มีค่า 0 มากกว่า 70%
print("\n🔧 Filtering out low-activity SKUs...")
sku_zero_pct = df.groupby('SKU')['Total_Qty'].apply(lambda x: (x == 0).sum() / len(x) * 100)
good_skus = sku_zero_pct[sku_zero_pct < 70].index
df = df[df['SKU'].isin(good_skus)].copy()
print(f"   Kept {len(good_skus)} SKUs (dropped {len(sku_zero_pct) - len(good_skus)} low-activity SKUs)")
print(f"   New dataset size: {len(df)} records")

# *** FEATURES LIST ***
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

# ตรวจสอบ features
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"\n⚠️  Missing features: {missing_features}")
    features = [f for f in features if f in df.columns]

NUM_FEATURES = len(features)
print(f"\nTotal features: {NUM_FEATURES}")

# --- 2. Time-based Split ---
print("\n=== Time-based Data Split ===")
df = df.sort_values(by=['SKU', 'Date'])

total_records = len(df)
train_end_idx = int(total_records * 0.7)
val_end_idx = int(total_records * 0.85)

df_train = df.iloc[:train_end_idx].copy()
df_val = df.iloc[train_end_idx:val_end_idx].copy()
df_test = df.iloc[val_end_idx:].copy()

print(f"Train: {len(df_train)} ({len(df_train)/total_records*100:.1f}%)")
print(f"       {df_train['Date'].min()} to {df_train['Date'].max()}")
print(f"Val:   {len(df_val)} ({len(df_val)/total_records*100:.1f}%)")
print(f"       {df_val['Date'].min()} to {df_val['Date'].max()}")
print(f"Test:  {len(df_test)} ({len(df_test)/total_records*100:.1f}%)")
print(f"       {df_test['Date'].min()} to {df_test['Date'].max()}")

# --- 3. Scaling (ใช้ RobustScaler แทน StandardScaler - ดีกว่าเมื่อมี outliers) ---
print("\n=== Scaling Features ===")
X_train_raw = df_train[features].values
y_train_raw = df_train[target].values

# ใช้ RobustScaler (ทนทานต่อ outliers กว่า)
scaler_X = RobustScaler()
scaler_X.fit(X_train_raw)

X_train_scaled = scaler_X.transform(X_train_raw)
X_val_scaled = scaler_X.transform(df_val[features].values)
X_test_scaled = scaler_X.transform(df_test[features].values)

print("✅ RobustScaler fitted on train set")

# Log Transform (เพิ่ม constant เล็กน้อยเพื่อจัดการกับ 0)
y_train_log = np.log1p(y_train_raw + 0.1)  # +0.1 เพื่อหลีกเลี่ยง log(0)
y_val_log = np.log1p(df_val[target].values + 0.1)
y_test_raw = df_test[target].values

# --- 4. Create Sequences Per SKU ---
print("\n=== Creating Sequences Per SKU ===")

def create_sequences_per_sku(X_scaled, y_log, df_subset, seq_length):
    all_X_seq = []
    all_y_seq = []
    
    for sku_code in df_subset['SKU'].unique():
        sku_indices = df_subset[df_subset['SKU'] == sku_code].index
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

print(f"X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}")
print(f"X_val:   {X_val_seq.shape}, y_val: {y_val_seq.shape}")
print(f"X_test:  {X_test_seq.shape}, y_test: {y_test_seq_raw.shape}")

if X_train_seq.shape[0] == 0:
    print("\n❌ ERROR: No sequences created!")
    print(f"   Try reducing SEQUENCE_LENGTH (current: {SEQUENCE_LENGTH})")
    exit(1)

# --- 5. Build and Train Model ---
input_shape = (SEQUENCE_LENGTH, NUM_FEATURES)
print(f"\n=== Building Model ===")
print(f"Input shape: {input_shape}")

demand_model = build_lstm_model(input_shape)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,  # เพิ่ม patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,  # เพิ่ม patience
    min_lr=1e-7,
    verbose=1
)

print("\n=== Training Model ===")
history = demand_model.fit(
    X_train_seq, y_train_seq,
    epochs=200,  # เพิ่ม epochs
    batch_size=16,  # ลด batch size (ช่วยเรียนรู้ได้ดีกว่า)
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

demand_model.save('demand_forecasting_model_improved.h5')
print("\n✅ Model saved!")

# --- 6. Evaluation ---
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Predict
predictions_log = demand_model.predict(X_test_seq).flatten()
predictions_raw = np.expm1(predictions_log) - 0.1  # ลบ 0.1 ที่เพิ่มไปตอนแรก
predictions_raw = np.maximum(0, predictions_raw)

# Metrics
mae = mean_absolute_error(y_test_seq_raw, predictions_raw)
mse = mean_squared_error(y_test_seq_raw, predictions_raw)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_seq_raw, predictions_raw)
mape = safe_mape(y_test_seq_raw, predictions_raw, epsilon=0.5)  # ใช้ safe MAPE

print(f"\n📊 Test Set Performance:")
print(f"{'Metric':<20} {'Value':<15}")
print("-" * 35)
print(f"{'MAE':<20} {mae:.2f} units")
print(f"{'RMSE':<20} {rmse:.2f} units")
print(f"{'R²':<20} {r2:.4f}")
print(f"{'MAPE (safe)':<20} {mape:.2f}%")
print("-" * 35)

# Interpretation
if r2 < 0.3:
    print("❌ R² < 0.3: Poor performance")
    print("\n💡 Possible reasons:")
    print("   1. Too many zero values in data")
    print("   2. Weak correlation between features and target")
    print("   3. Need more data or better features")
elif r2 < 0.5:
    print("⚠️  R² < 0.5: Moderate performance")
elif r2 < 0.7:
    print("⚠️  R² < 0.7: Good performance")
else:
    print("✅ R² ≥ 0.7: Excellent performance!")

print(f"\n📈 Statistics:")
print(f"{'Statistic':<20} {'Actual':<15} {'Predicted':<15}")
print("-" * 50)
print(f"{'Mean':<20} {y_test_seq_raw.mean():.2f} {predictions_raw.mean():.2f}")
print(f"{'Std':<20} {y_test_seq_raw.std():.2f} {predictions_raw.std():.2f}")
print(f"{'Min':<20} {y_test_seq_raw.min():.2f} {predictions_raw.min():.2f}")
print(f"{'Max':<20} {y_test_seq_raw.max():.2f} {predictions_raw.max():.2f}")

# --- 7. Visualization (แก้ path สำหรับ Windows) ---
try:
    output_dir = "."  # บันทึกใน current directory
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', alpha=0.8)
    plt.plot(history.history['val_loss'], label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.scatter(y_test_seq_raw, predictions_raw, alpha=0.5, s=20)
    plt.plot([y_test_seq_raw.min(), y_test_seq_raw.max()],
             [y_test_seq_raw.min(), y_test_seq_raw.max()], 'r--', lw=2)
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title(f'Actual vs Predicted (R²={r2:.4f})')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    residuals = y_test_seq_raw - predictions_raw
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # บันทึกใน current directory
    save_path = os.path.join(output_dir, 'model_evaluation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {save_path}")
    plt.close()
    
except Exception as e:
    print(f"\n⚠️  Could not create visualization: {e}")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED!")
print("=" * 70)