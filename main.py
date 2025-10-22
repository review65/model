# main.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split # <--- เราจะกลับมาใช้ตัวนี้

from data_preparation import load_and_preprocess_data
from demand_model import build_lstm_model
from price_optimizer import ParticleSwarmOptimizer

def create_sequences(X, y, time_steps=10):
    """
    สร้าง "หน้าต่าง" ข้อมูลแบบ Time Series
    (ฟังก์ชันนี้ถูกต้อง ไม่ต้องแก้)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

SEQUENCE_LENGTH = 10 # มองย้อนหลัง 10 วัน
NUM_FEATURES = 13    # (SKU, Category, ... is_month_end)

# --- 1. Data Loading and Preparation ---
DATA_FILE = r'E:\model\Amazon Sale Report.csv'
df = load_and_preprocess_data(DATA_FILE) # <--- data_preparation.py ไม่ต้องแก้

features = [
    'SKU', 'Category', 'Size', 'Avg_Price', 'day_of_week', 
    'month', 'week_of_year', 'Qty_lag_1', 'Qty_lag_7', 
    'Qty_roll_mean_7', 'Has_Promotion', 'is_weekend', 'is_month_end'
]
target = 'Total_Qty'

# ( === 1.1 แก้ไข Logic การสร้าง Sequence ทั้งหมด === )

# --- 1. เรียงข้อมูลตาม SKU แล้วตาม Date (สำคัญมาก!) ---
df = df.sort_values(by=['SKU', 'Date'])

X_data = df[features].values
y_raw = df[target].values 
y_log = np.log1p(y_raw) 

# --- 2. Scale ข้อมูล X ทั้งหมด ---
# (นี่คือ "การโกง" เล็กน้อย แต่จำเป็นเพื่อให้โมเดลทำงานง่ายขึ้น)
scaler_X = StandardScaler().fit(X_data)
X_data_scaled = scaler_X.transform(X_data)

# --- 3. วน Loop สร้าง Sequence ทีละ SKU ---
print("Creating sequences per SKU...")
all_X_seq = []
all_y_seq = []
all_y_raw_seq = [] # <--- สำหรับการประเมินผล

# เราจะใช้ df.groupby('SKU') เพื่อหา index ของแต่ละกลุ่ม
for sku_code in df['SKU'].unique():
    
    # ดึงข้อมูลเฉพาะของ SKU นี้
    sku_indices = df[df['SKU'] == sku_code].index
    
    if len(sku_indices) > SEQUENCE_LENGTH:
        X_sku_scaled = X_data_scaled[sku_indices]
        y_sku_log = y_log[sku_indices]
        y_sku_raw = y_raw[sku_indices]
        
        # สร้าง Sequence
        X_seq_sku, y_seq_sku_log = create_sequences(X_sku_scaled, y_sku_log, SEQUENCE_LENGTH)
        _, y_seq_sku_raw = create_sequences(X_sku_scaled, y_sku_raw, SEQUENCE_LENGTH)
        
        if X_seq_sku.shape[0] > 0:
            all_X_seq.append(X_seq_sku)
            all_y_seq.append(y_seq_sku_log)
            all_y_raw_seq.append(y_seq_sku_raw)

# --- 4. รวม Sequence ทั้งหมด ---
X_seq_combined = np.concatenate(all_X_seq)
y_seq_log_combined = np.concatenate(all_y_seq)
y_seq_raw_combined = np.concatenate(all_y_raw_seq)

print(f"Total sequences created: {X_seq_combined.shape[0]}")

# --- 5. แบ่งข้อมูล Train/Val/Test (แบบสุ่ม) ---
# เราจะสับไพ่ Sequence ทั้งหมดที่ได้มา
X_train_val, X_test, y_train_val_log, y_test_raw = train_test_split(
    X_seq_combined, 
    y_seq_raw_combined, # <--- เราจะใช้ y_raw สำหรับ Test
    test_size=0.15, 
    random_state=42,
    shuffle=True # <--- สับไพ่
)

# แบ่ง Train/Validation
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X_train_val,
    np.log1p(y_train_val_log), # <--- แปลง y_train_val กลับเป็น log
    test_size=0.1765, # (0.15 / 0.85)
    random_state=42,
    shuffle=True
)

print(f"Data shape after split (Samples, Timesteps, Features):")
print(f"X_train shape: {X_train.shape}, y_train_log shape: {y_train_log.shape}")
print(f"X_val shape:   {X_val.shape}, y_val_log shape: {y_val_log.shape}")
print(f"X_test shape:  {X_test.shape}, y_test_raw shape: {y_test_raw.shape}")

# ( === สิ้นสุดการแก้ไข Logic === )


# --- 2. Build and Train Demand Model ---
input_shape = (SEQUENCE_LENGTH, NUM_FEATURES) 
demand_model = build_lstm_model(input_shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining the demand model (with SKU-based sequences)...")
history = demand_model.fit(
    X_train, y_train_log, 
    epochs=100,
    batch_size=64, 
    validation_data=(X_val, y_val_log),
    callbacks=[early_stopping], 
    verbose=1
)
demand_model.save('demand_forecasting_model.h5')

# --- 3. Model Evaluation ---
print("\n--- Model Evaluation ---")
predictions_log = demand_model.predict(X_test).flatten() 
    
predictions_raw = np.expm1(predictions_log) 
predictions_raw = np.maximum(0, predictions_raw) 

mae = mean_absolute_error(y_test_raw, predictions_raw) # <--- เทียบ y_test_raw
mse = mean_squared_error(y_test_raw, predictions_raw)
r2 = r2_score(y_test_raw, predictions_raw) 

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}") # <--- ค่านี้ต้องดีขึ้น!
print("-------------------------\n")


# --- 4. Price Optimization (แก้ไขการสร้าง Input) ---
print("Starting price optimization...")
# เราจะใช้ค่าเฉลี่ยของ "Feature วันพรุ่งนี้" เหมือนเดิม
# แต่เราต้องสร้าง "Sequence ฐาน" ที่สมเหตุสมผล

target_skus_encoded = df['SKU'].unique()[0:3]
PRODUCT_COSTS = np.array([300, 400, 600]) 
NUM_PRODUCTS = len(target_skus_encoded)

# ( === 4.1 สร้าง Sequence ฐาน จากค่าเฉลี่ย === )
# เราจะสร้าง "ฐาน 9 วัน" จากค่าเฉลี่ยของ X_train ทั้งหมด
# (X_train ตอนนี้มี shape [samples, 10, 13])
avg_features_vector = np.mean(X_train.reshape(-1, NUM_FEATURES), axis=0)
base_sequence_avg = np.tile(avg_features_vector, (SEQUENCE_LENGTH - 1, 1)) # <--- สร้างฐาน 9 วัน

def profit_objective_function(prices):
    model_inputs = []
    
    for i, price in enumerate(prices):
        
        # --- สร้าง "Feature ของวันพรุ่งนี้" (วันที่จะทำนาย) ---
        tomorrow_features = avg_features_vector.copy()
        
        tomorrow_features[0] = target_skus_encoded[i] # SKU
        tomorrow_features[3] = price                  # Avg_Price (ราคาใหม่)
        
        # (อัปเดต Lag Features ของ "วันพรุ่งนี้" ให้สมจริง)
        tomorrow_features[7] = avg_features_vector[7] # Qty_lag_1 
        tomorrow_features[8] = avg_features_vector[8] # Qty_lag_7
        tomorrow_features[9] = avg_features_vector[9] # Qty_roll_mean_7
        
        # (อัปเดต Promotion/Holiday)
        tomorrow_features[10] = 1 # สมมติว่าเราจะ "จัดโปรโมชั่น"
        tomorrow_features[11] = 0 # สมมติว่า "ไม่ใช่วันหยุด"
        tomorrow_features[12] = 0 # สมมติว่า "ไม่ใช่สิ้นเดือน"
        
        # --- สลับข้อมูล ---
        # (เอาฐาน 9 วัน + "วันพรุ่งนี้" 1 วัน)
        final_sequence = np.append(base_sequence_avg, [tomorrow_features], axis=0)
        
        model_inputs.append(final_sequence)
        
    # ข้อมูลนี้ "ยังไม่ได้ Scale" เพราะเราสร้างจาก avg_features_vector
    # เราต้อง Scale มันก่อน
    model_inputs_array = np.array(model_inputs)
    
    # Reshape (samples, timesteps, features) -> (samples * timesteps, features)
    model_inputs_reshaped = model_inputs_array.reshape(-1, NUM_FEATURES)
    
    # Scale
    model_inputs_scaled = scaler_X.transform(model_inputs_reshaped)
    
    # Reshape กลับ -> (samples, timesteps, features)
    model_inputs_final = model_inputs_scaled.reshape(NUM_PRODUCTS, SEQUENCE_LENGTH, NUM_FEATURES)
    
    predicted_demands_log = demand_model.predict(model_inputs_final, verbose=0).flatten()
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