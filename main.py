# main.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from data_preparation import load_and_preprocess_data
from demand_model import build_lstm_model
from price_optimizer import ParticleSwarmOptimizer

# ( === 1. สร้าง Helper Function สำหรับสร้าง Sequence === )
def create_sequences(X, y, time_steps=10):
    """
    สร้าง "หน้าต่าง" ข้อมูลแบบ Time Series
    X: ข้อมูลย้อนหลัง time_steps วัน
    y: ข้อมูลวันที่ time_steps + 1
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# ( === 2. กำหนดค่า Sequence === )
SEQUENCE_LENGTH = 10 # เราจะมองย้อนหลัง 10 วัน เพื่อทำนายวันที่ 11

# --- 1. Data Loading and Preparation ---
DATA_FILE = r'E:\model\Amazon Sale Report.csv'
df = load_and_preprocess_data(DATA_FILE)

features = [
    'SKU', 'Category', 'Size', 'Avg_Price', 'day_of_week', 
    'month', 'week_of_year', 'Qty_lag_1', 'Qty_lag_7', 
    'Qty_roll_mean_7', 'Has_Promotion', 'is_weekend', 'is_month_end'
]
# (เรามี 13 features)
NUM_FEATURES = len(features) 
target = 'Total_Qty'

# --- 1.1 จัดการข้อมูลสำหรับ Sequence ---
print("\nSplitting data based on time...")
df = df.sort_values(by='Date')

X_data = df[features].values
y_raw = df[target].values 
y_log = np.log1p(y_raw) 

# --- 1.2 แบ่งข้อมูล Train/Validation/Test (ก่อนสร้าง Sequence) ---
train_ratio = 0.7
val_ratio = 0.15

train_index = int(len(df) * train_ratio)
val_index = int(len(df) * (train_ratio + val_ratio))

X_train, X_val, X_test = X_data[:train_index], X_data[train_index:val_index], X_data[val_index:]
y_train_log, y_val_log = y_log[:train_index], y_log[train_index:val_index]
y_test_raw = y_raw[val_index:]

print(f"Data split by time (before sequencing): Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# --- 1.3 ปรับสเกลข้อมูล (Fit เฉพาะกับ Train) ---
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val) 
X_test_scaled = scaler_X.transform(X_test)

# ( === 1.4 สร้าง Sequence! (นี่คือส่วนที่เปลี่ยนไป) === )
print("Creating time sequences...")
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_log, SEQUENCE_LENGTH)
X_val_seq, y_val_seq     = create_sequences(X_val_scaled, y_val_log, SEQUENCE_LENGTH)
X_test_seq, y_test_seq   = create_sequences(X_test_scaled, y_test_raw, SEQUENCE_LENGTH)

print(f"Data shape after sequencing (Samples, Timesteps, Features):")
print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"X_val_seq shape:   {X_val_seq.shape}")
print(f"X_test_seq shape:  {X_test_seq.shape}")

# --- 2. Build and Train Demand Model ---
# ( === 2.1 อัปเดต Input Shape === )
# Shape ใหม่คือ (10, 13)
input_shape = (SEQUENCE_LENGTH, NUM_FEATURES) 
demand_model = build_lstm_model(input_shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining the demand model (with sequences)...")
history = demand_model.fit(
    X_train_seq, y_train_seq, # <--- ใช้ข้อมูล Sequence
    epochs=100,
    batch_size=64, 
    validation_data=(X_val_seq, y_val_seq), # <--- ใช้ข้อมูล Sequence
    callbacks=[early_stopping], 
    verbose=1
)
demand_model.save('demand_forecasting_model.h5')

# --- 3. Model Evaluation ---
print("\n--- Model Evaluation ---")
predictions_log = demand_model.predict(X_test_seq).flatten() # <--- ใช้ข้อมูล Sequence
    
predictions_raw = np.expm1(predictions_log) 
predictions_raw = np.maximum(0, predictions_raw) 

# (y_test_seq คือ y_raw ที่ถูกหั่นเป็น Sequence แล้ว)
mae = mean_absolute_error(y_test_seq, predictions_raw) 
mse = mean_squared_error(y_test_seq, predictions_raw)
r2 = r2_score(y_test_seq, predictions_raw) 

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}") # <--- หวังว่าค่านี้จะพุ่ง!
print("-------------------------\n")


# --- 4. Price Optimization (แก้ไขการสร้าง Input) ---
print("Starting price optimization...")
# (ส่วนนี้จะซับซ้อนขึ้นเล็กน้อย เพราะเราต้องสร้าง "Sequence จำลอง")

target_skus_encoded = df['SKU'].unique()[0:3]
PRODUCT_COSTS = np.array([300, 400, 600]) 
NUM_PRODUCTS = len(target_skus_encoded)

# ( === 4.1 ดึงข้อมูล 10 วันล่าสุด (scaled) มาเป็นฐาน === )
# เราจะใช้ข้อมูล 10 วันสุดท้ายจาก X_val_scaled เป็น "ฐาน" ในการทำนาย
base_sequence = X_val_scaled[-SEQUENCE_LENGTH:]

def profit_objective_function(prices):
    model_inputs = []
    
    # ดึงค่าเฉลี่ยของ Feature ทั้งหมดจาก X_train (สำหรับ Feature ที่เราไม่ได้เปลี่ยน)
    avg_features_vector = np.mean(X_train, axis=0)
    
    for i, price in enumerate(prices):
        
        # สร้าง Sequence ใหม่ โดยใช้ 10 วันล่าสุดเป็นฐาน
        new_sequence = base_sequence.copy()
        
        # --- สร้าง "Feature ของวันพรุ่งนี้" (วันที่จะทำนาย) ---
        # เราจะใช้ค่าเฉลี่ยของ Feature ส่วนใหญ่
        tomorrow_features = avg_features_vector.copy()
        
        tomorrow_features[0] = target_skus_encoded[i] # SKU
        tomorrow_features[3] = price                  # Avg_Price (ราคาใหม่)
        
        # (อัปเดต Lag Features ของ "วันพรุ่งนี้" ให้สมจริง)
        # เราจะสมมติว่ายอดขาย "เมื่อวาน" (lag 1) คือค่าเฉลี่ย
        tomorrow_features[7] = avg_features_vector[7] # Qty_lag_1 
        tomorrow_features[8] = avg_features_vector[8] # Qty_lag_7
        tomorrow_features[9] = avg_features_vector[9] # Qty_roll_mean_7
        
        # (อัปเดต Promotion/Holiday)
        tomorrow_features[10] = 1 # สมมติว่าเราจะ "จัดโปรโมชั่น"
        tomorrow_features[11] = 0 # สมมติว่า "ไม่ใช่วันหยุด"
        tomorrow_features[12] = 0 # สมมติว่า "ไม่ใช่สิ้นเดือน"
        
        # --- สลับข้อมูล ---
        # เขยิบ Sequence (ลบวันเก่าสุด, เพิ่ม "วันพรุ่งนี้")
        final_sequence = np.append(new_sequence[1:], [tomorrow_features], axis=0)
        
        model_inputs.append(final_sequence)
        
    model_inputs_scaled = np.array(model_inputs) # (ข้อมูลถูก Scaled แล้ว ไม่ต้อง scale ซ้ำ)
    # Shape ที่ได้คือ (3, 10, 13)
    
    predicted_demands_log = demand_model.predict(model_inputs_scaled, verbose=0).flatten()
    predicted_demands_raw = np.expm1(predicted_demands_log)
    predicted_demands_raw = np.maximum(0, predicted_demands_raw)
    
    total_profit = np.sum((prices - PRODUCT_COSTS) * predicted_demands_raw)
    
    return -total_profit

# ( ... โค้ดส่วนที่เหลือเหมือนเดิม ... )
price_bounds = [
    (400, 800),   
    (500, 1000),  
    (700, 1500)  
]
optimizer = ParticleSwarmOptimizer(
    objective_function=profit_objective_function,
    bounds=price_bounds,
    num_particles=40,
    max_iter=50
)
optimal_prices, max_profit = optimizer.optimize()

print(f"\nOptimal Prices Found: {np.round(optimal_prices, 2)}")
print(f"Maximum Estimated Profit: {-max_profit:.2f}")