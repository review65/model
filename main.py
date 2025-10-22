# main.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

from data_preparation import load_and_preprocess_data
from demand_model import build_lstm_model
from price_optimizer import ParticleSwarmOptimizer

# --- 1. Data Loading and Preparation ---
DATA_FILE = 'Amazon Sale Report.csv'
df = load_and_preprocess_data(DATA_FILE)

# กำหนด features (X) และ target (y)
features =
target = 'Total_Qty'

X = df[features].values
y = df[target].values

# แบ่งข้อมูล (ควรแบ่งตามเวลาสำหรับ Time Series แต่ในที่นี้ใช้ split แบบสุ่มเพื่อความง่าย)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับสเกลข้อมูล
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Reshape ข้อมูลสำหรับ LSTM [samples, timesteps, features]
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape, 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape, 1, X_test_scaled.shape[1]))

# --- 2. Build and Train Demand Model ---
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
demand_model = build_lstm_model(input_shape)

print("\nTraining the demand model...")
history = demand_model.fit(X_train_reshaped, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=1)
demand_model.save('demand_forecasting_model.h5')

# --- 3. Model Evaluation (ส่วนที่เพิ่มเข้ามา) ---
print("\n--- Model Evaluation ---")
predictions = demand_model.predict(X_test_reshaped).flatten()

# คำนวณค่า Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")
print("-------------------------\n")


# --- 4. Price Optimization ---
print("Starting price optimization...")

# สมมติว่าเราต้องการหาราคาสำหรับสินค้า 3 ชนิด (เลือก SKU ที่มีข้อมูลเยอะ)
# คุณต้องแทนที่ค่าเหล่านี้ด้วย SKU และต้นทุนจริงของคุณ
target_skus_encoded =.astype('category').cat.codes, 
                       df.astype('category').cat.codes[1], 
                       df.astype('category').cat.codes[2]] 
PRODUCT_COSTS = np.array() # ต้นทุนสมมติ
NUM_PRODUCTS = len(target_skus_encoded)

def profit_objective_function(prices):
    """
    ฟังก์ชันคำนวณกำไร (ที่เราจะหาค่าต่ำสุดของ -profit)
    """
    model_inputs =
    
    # สร้าง input สำหรับแต่ละราคาสินค้า
    for i, price in enumerate(prices):
        # ใช้ค่าเฉลี่ยของ features อื่นๆ จากข้อมูล training
        # ยกเว้น SKU และ Price
        avg_features = np.mean(X_train[:, [1, 2, 3, 4, 5]], axis=0)
        
        # สร้าง feature vector:
        current_features = np.concatenate(([target_skus_encoded[i]], avg_features[:2], [price], avg_features[2:]))
        model_inputs.append(current_features)

    model_inputs_scaled = scaler_X.transform(np.array(model_inputs))
    model_inputs_reshaped = model_inputs_scaled.reshape((NUM_PRODUCTS, 1, len(features)))
    
    predicted_demands = demand_model.predict(model_inputs_reshaped, verbose=0).flatten()
    
    # ป้องกันค่าพยากรณ์ติดลบ
    predicted_demands = np.maximum(0, predicted_demands)
    
    total_profit = np.sum((prices - PRODUCT_COSTS) * predicted_demands)
    
    return -total_profit

# กำหนดขอบเขตของราคาสำหรับสินค้าแต่ละชนิด
price_bounds = [
    (400, 800),   # ขอบเขตราคาสำหรับสินค้าชิ้นที่ 1
    (500, 1000),  # ขอบเขตราคาสำหรับสินค้าชิ้นที่ 2
    (700, 1500)   # ขอบเขตราคาสำหรับสินค้าชิ้นที่ 3
]

# รัน PSO
optimizer = ParticleSwarmOptimizer(
    objective_function=profit_objective_function,
    bounds=price_bounds,
    num_particles=40,
    max_iter=50
)

optimal_prices, max_profit = optimizer.optimize()

print(f"\nOptimal Prices Found: {np.round(optimal_prices, 2)}")
print(f"Maximum Estimated Profit: {max_profit:.2f}")