# main.py
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split # <--- ไม่ใช้แล้ว
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # <--- Import เข้ามาใหม่

from data_preparation import load_and_preprocess_data
from demand_model import build_lstm_model
from price_optimizer import ParticleSwarmOptimizer

# --- 1. Data Loading and Preparation ---
DATA_FILE = r'E:\model\Amazon Sale Report.csv' # <--- แก้ Path ให้คุณตามครั้งก่อน
df = load_and_preprocess_data(DATA_FILE)

# ( === ส่วนที่แก้ไขเพื่อเพิ่ม Lag Features === )
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

# ( === ส่วนที่แก้ไขการแบ่งข้อมูลทั้งหมด === )

print("\nSplitting data based on time...")
# !!! สำคัญมาก: ต้องเรียงข้อมูลตามวันท
df = df.sort_values(by='Date')

X = df[features].values
y_raw = df[target].values

y = np.log1p(y_raw)

# --- แบ่งข้อมูล 3 ส่วนตามเวลา (Train 70%, Validation 15%, Test 15%) ---
train_ratio = 0.7
val_ratio = 0.15

train_index = int(len(df) * train_ratio)
val_index = int(len(df) * (train_ratio + val_ratio))

X_train, y_train = X[:train_index], y[:train_index]
X_val, y_val     = X[train_index:val_index], y[train_index:val_index]
X_test, y_test   = X[val_index:], y[val_index:]

print(f"Data split by time: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# --- ปรับสเกลข้อมูล (Fit เฉพาะกับข้อมูล Train เท่านั้น) ---
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val) # <--- แปลง Validation Set
X_test_scaled = scaler_X.transform(X_test)

# Reshape ข้อมูลสำหรับ LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1])) # <--- Reshape Validation Set
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ( === สิ้นสุดส่วนที่แก้ไขการแบ่งข้อมูล === )


# --- 2. Build and Train Demand Model ---
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
demand_model = build_lstm_model(input_shape)

# --- เพิ่ม Early Stopping ---
# (หยุดเทรนอัตโนมัติเมื่อ val_loss ไม่ดีขึ้น 10 รอบติดต่อกัน และคืนค่าน้ำหนักที่ดีที่สุด)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining the demand model...")
history = demand_model.fit(
    X_train_reshaped, y_train, 
    epochs=100,  # <--- เพิ่ม Epochs (เช่น 100)
    batch_size=64, 
    validation_data=(X_val_reshaped, y_val), # <--- ใช้ Validation Set ที่เราแบ่งเอง
    callbacks=[early_stopping], # <--- ใส่ EarlyStopping
    verbose=1
)
demand_model.save('demand_forecasting_model.h5')

# --- 3. Model Evaluation ---
print("\n--- Model Evaluation ---")
predictions_log = demand_model.predict(X_test_reshaped).flatten()# <--- ตอนนี้จะทดสอบกับ Test Set ที่ถูกต้อง

# --- แปลงค่าที่ทำนายได้ กลับเป็นค่าเดิม ---
predictions = np.expm1(predictions_log)
# ป้องกันค่าติดลบ (เผื่อไว้)
predictions = np.maximum(0, predictions)

# คำนวณค่า Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}") # <--- ค่านี้ควรจะสูงขึ้นมาก
print("-------------------------\n")


# --- 4. Price Optimization ---
print("Starting price optimization...")

target_skus_encoded = df['SKU'].unique()[0:3]
PRODUCT_COSTS = np.array([300, 400, 600]) 
NUM_PRODUCTS = len(target_skus_encoded)

def profit_objective_function(prices):
    """
    ฟังก์ชันคำนวณกำไร (ที่เราจะหาค่าต่ำสุดของ -profit)
    """
    model_inputs = []
    
    # <--- Logic ส่วนนี้ยังใช้ได้ แต่จะแม่นยำขึ้นเพราะ scaler_X fit ถูกต้องแล้ว
    # (ใช้ค่าเฉลี่ยจาก X_train ที่ถูกต้อง)
    avg_features_vector = np.mean(X_train, axis=0)
    
    for i, price in enumerate(prices):
        current_features = avg_features_vector.copy()
        
        # แทนที่ 'SKU' (index 0) 
        current_features[0] = target_skus_encoded[i]
        
        # แทนที่ 'Avg_Price' (index 3) 
        current_features[3] = price
        
        # <--- เพิ่มการแทนที่ Lag Features (index 7, 8, 9)
        # เราจะใช้ค่าเฉลี่ยของ Lag Features จาก X_train เป็นฐาน
        # (เพื่อให้การทำนายราคาในอนาคตมีความสมเหตุสมผล)
        current_features[7] = avg_features_vector[7] # Qty_lag_1
        current_features[8] = avg_features_vector[8] # Qty_lag_7
        current_features[9] = avg_features_vector[9] # Qty_roll_mean_7
        
        model_inputs.append(current_features)

    model_inputs_scaled = scaler_X.transform(np.array(model_inputs))
    model_inputs_reshaped = model_inputs_scaled.reshape((NUM_PRODUCTS, 1, len(features)))
    
    predicted_demands = demand_model.predict(model_inputs_reshaped, verbose=0).flatten()
    predicted_demands = np.maximum(0, predicted_demands)
    total_profit = np.sum((prices - PRODUCT_COSTS) * predicted_demands)
    
    return -total_profit

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

# ( === แก้ไขการแสดงผลกำไรตามที่เราคุยกันครั้งก่อน === )
print(f"\nOptimal Prices Found: {np.round(optimal_prices, 2)}")
print(f"Maximum Estimated Profit: {-max_profit:.2f}") # <--- เติมเครื่องหมายลบ