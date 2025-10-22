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

# --- 1. Data Loading and Preparation ---
DATA_FILE = r'E:\model\Amazon Sale Report.csv'
df = load_and_preprocess_data(DATA_FILE)

# ( === นี่คือ List Feature ที่สมบูรณ์ === )
features = [
    'SKU', 
    'Category', 
    'Size', 
    'Avg_Price', 
    'day_of_week', 
    'month', 
    'week_of_year',
    'Qty_lag_1',       
    'Qty_lag_7',       
    'Qty_roll_mean_7',  
    'Has_Promotion',    
    'is_weekend',       
    'is_month_end'      
]
target = 'Total_Qty'

# ( === ส่วนที่แก้ไข Log Transform ทั้งหมด === )

print("\nSplitting data based on time...")
df = df.sort_values(by='Date')

X = df[features].values
y_raw = df[target].values # <--- 1. เก็บค่า Y ดั้งเดิม (Raw)

# --- 2. แปลง Y เป็น Log ---
y_log = np.log1p(y_raw) # log1p คือ log(1 + x)

# --- 3. แบ่งข้อมูล 3 ส่วนตามเวลา ---
train_ratio = 0.7
val_ratio = 0.15

train_index = int(len(df) * train_ratio)
val_index = int(len(df) * (train_ratio + val_ratio))

# --- แบ่ง X ---
X_train, X_val, X_test = X[:train_index], X[train_index:val_index], X[val_index:]

# --- แบ่ง Y (สำคัญมาก) ---
y_train_log = y_log[:train_index]         # <--- Y สำหรับเทรน (Log)
y_val_log   = y_log[train_index:val_index] # <--- Y สำหรับ Validate (Log)
y_test_raw  = y_raw[val_index:]          # <--- Y สำหรับ Test (Raw) !!!

print(f"Data split by time: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# --- ปรับสเกลข้อมูล X (Fit เฉพาะกับข้อมูล Train เท่านั้น) ---
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val) 
X_test_scaled = scaler_X.transform(X_test)

# Reshape ข้อมูล X สำหรับ LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ( === สิ้นสุดส่วนที่แก้ไข Log Transform === )


# --- 2. Build and Train Demand Model ---
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
demand_model = build_lstm_model(input_shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining the demand model...")
history = demand_model.fit(
    X_train_reshaped, y_train_log,  # <--- เทรนด้วย y_train_log
    epochs=100,
    batch_size=64, 
    validation_data=(X_val_reshaped, y_val_log), # <--- Validate ด้วย y_val_log
    callbacks=[early_stopping], 
    verbose=1
)
demand_model.save('demand_forecasting_model.h5')

# --- 3. Model Evaluation (แก้ไขการเปรียบเทียบ) ---
print("\n--- Model Evaluation ---")
predictions_log = demand_model.predict(X_test_reshaped).flatten() # <--- 1. ผลลัพธ์คือค่า Log
    
# --- 2. แปลงค่าที่ทำนายได้ กลับเป็นค่าเดิม ---
predictions_raw = np.expm1(predictions_log) # expm1 คือ e^x - 1
predictions_raw = np.maximum(0, predictions_raw) # ป้องกันค่าติดลบ

# --- 3. คำนวณค่า Metrics (เทียบ "ของจริง" (Raw) กับ "ของจริง" (Raw)) ---
mae = mean_absolute_error(y_test_raw, predictions_raw) # <--- y_test_raw vs predictions_raw
mse = mean_squared_error(y_test_raw, predictions_raw)
r2 = r2_score(y_test_raw, predictions_raw) # <--- y_test_raw vs predictions_raw

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}") # <--- ค่านี้จะสะท้อนความจริง
print("-------------------------\n")


# --- 4. Price Optimization (แก้ไขการสร้าง Feature ให้ครบ) ---
print("Starting price optimization...")

target_skus_encoded = df['SKU'].unique()[0:3]
PRODUCT_COSTS = np.array([300, 400, 600]) 
NUM_PRODUCTS = len(target_skus_encoded)

def profit_objective_function(prices):
    model_inputs = []
    avg_features_vector = np.mean(X_train, axis=0)
    
    for i, price in enumerate(prices):
        current_features = avg_features_vector.copy()
        
        # ( === แก้ไข Index ให้ครบ 13 Features === )
        current_features[0] = target_skus_encoded[i] # SKU
        current_features[3] = price                  # Avg_Price
        
        # เราจะใช้ค่าเฉลี่ยของ Feature อื่นๆ ที่เหลือจาก X_train เป็นฐาน
        current_features[7] = avg_features_vector[7] # Qty_lag_1
        current_features[8] = avg_features_vector[8] # Qty_lag_7
        current_features[9] = avg_features_vector[9] # Qty_roll_mean_7
        current_features[10] = avg_features_vector[10] # Has_Promotion
        current_features[11] = avg_features_vector[11] # is_weekend
        current_features[12] = avg_features_vector[12] # is_month_end
        
        model_inputs.append(current_features)

    model_inputs_scaled = scaler_X.transform(np.array(model_inputs))
    model_inputs_reshaped = model_inputs_scaled.reshape((NUM_PRODUCTS, 1, len(features)))
    
    # --- การทำนายกำไรต้องใช้ Log Transform ด้วย ---
    predicted_demands_log = demand_model.predict(model_inputs_reshaped, verbose=0).flatten()
    predicted_demands_raw = np.expm1(predicted_demands_log) # <--- แปลงกลับเป็นยอดขายจริง
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