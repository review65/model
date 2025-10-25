# main.py (WEEKLY - COMPARE 3 MODELS + OPTIMIZE 3 MODELS)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import functools # (Import functools)

# --- !! 1. IMPORT MODELS FOR COMPARISON !! ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from aggregate_weekly import aggregate_data
from price_optimizer import ParticleSwarmOptimizer # <-- เปิดใช้งาน

# (ส่วน 3, 4, 5, 6, 7 เหมือนเดิมเป๊ะ... ผมย่อไว้)

# --- 3. LOAD WEEKLY DATA ---
DATA_FILE = r'E:\model\model\Amazon Sale Report.csv'
print("\n=== Loading and Aggregating Weekly Data ===")
df = aggregate_data(DATA_FILE)

# --- 4. ENCODE CATEGORICAL FEATURES ---
print("\nEncoding categorical features (SKU, Category, Size)...")
encoders = {}
for col in ['SKU', 'Category', 'Size']:
    le = LabelEncoder()
    # (แก้ปัญหา SKU ใหม่ที่อาจเจอใน test set)
    df[col] = le.fit_transform(df[col].astype(str)) 
    encoders[col] = le

# --- 5. NEW WEEKLY FEATURE ENGINEERING ---
print("\n=== Creating WEEKLY Features ===")
df = df.sort_values(by=['SKU', 'Date'])
# Lag Features
df['Qty_lag_1'] = df.groupby('SKU')['Total_Qty'].shift(1)
df['Price_lag_1'] = df.groupby('SKU')['Avg_Price'].shift(1)
df['price_change_pct'] = (df['Avg_Price'] - df['Price_lag_1']) / (df['Price_lag_1'] + 1e-6)
# Time Features
df = df.reset_index()
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(float)
df['month'] = df['Date'].dt.month.astype(float)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
print(f"Original shape: {df.shape}")
df = df.dropna()
print(f"Shape after dropping NaNs: {df.shape}")

# --- 6. NEW FEATURE LIST (Weekly) ---
features = [
    'SKU', 'Category', 'Size', 'Avg_Price', 'Max_Price', 'Min_Price',
    'week_of_year', 'month', 'month_sin', 'month_cos',
    'Qty_lag_1',
    'Price_lag_1', 'price_change_pct',
    'Has_Promotion' # (ยืนยันว่า Transaction_Count ออกแล้ว)
]
target = 'Total_Qty'
NUM_FEATURES = len(features)
print(f"\nTotal weekly features: {NUM_FEATURES}")

# --- 7. Time-based Train/Val/Test Split ---
print("\n=== Time-based Data Split ===")
print("Sorting by Date first for proper time-based split...")
df = df.sort_values(by=['Date'])
total_records = len(df)
train_end_idx = int(total_records * 0.7)
val_end_idx = int(total_records * 0.85) 
df_train = df.iloc[:train_end_idx].copy()
df_val = df.iloc[train_end_idx:val_end_idx].copy()
df_test = df.iloc[val_end_idx:].copy()

print(f"Train period: {df_train['Date'].min()} to {df_train['Date'].max()}")
print(f"Val period:   {df_val['Date'].min()} to {df_val['Date'].max()}")
print(f"Test period:  {df_test['Date'].min()} to {df_test['Date'].max()}")

# --- 8. Fit Scaler ---
print("\n=== Fitting Scaler on Train Set ONLY ===")
X_train_raw = df_train[features].values
y_train_target = df_train[target].values
X_test_raw = df_test[features].values
y_test_raw = df_test[target].values

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

print(f"X_train shape (flat): {X_train_scaled.shape}")
print(f"X_test shape (flat): {X_test_scaled.shape}")

# --- 9. Build, Train, and Compare Models ---
print("\n=== Training and Comparing Models ===")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10, 
        random_state=42,
        n_jobs=-1
    ),
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        random_state=42,
        max_iter=500,
        early_stopping=True
    )
}

results = {}
best_model_name = None
best_r2 = -np.inf

print(f"\nTest Set Performance (Actual Mean: {y_test_raw.mean():.2f} units)")
print("-" * 50)

for name, model in models.items():
    print(f"Training: {name}...")
    model.fit(X_train_scaled, y_train_target)
    predictions_raw = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    
    # (เก็บโมเดลที่เทรนแล้วไว้)
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'model_obj': model, 'preds': predictions_raw}
    
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.2f} units")
    print(f"  RMSE: {rmse:.2f} units")
    print("-" * 50)
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name

print(f"\n🏆 Best Model: {best_model_name} (Based on R²) 🏆")

# --- 10. Visualization (Plotting the BEST model) ---
best_model_preds = results[best_model_name]['preds']
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test_raw, best_model_preds, alpha=0.5)
plt.plot([y_test_raw.min(), y_test_raw.max()],
         [y_test_raw.min(), y_test_raw.max()], 'r--', lw=2)
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title(f'Actual vs Predicted ({best_model_name})\n(R²={results[best_model_name]["R²"]:.4f})')
plt.grid(True)
plt.subplot(1, 2, 2)
residuals = y_test_raw - best_model_preds
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title(f'Residual Distribution ({best_model_name})')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'model_evaluation_weekly_COMPARISON.png', dpi=150)
print(f"\nVisualization for best model ({best_model_name}) saved to: model_evaluation_weekly_COMPARISON.png")


# --- !! 11. PRICE OPTIMIZATION SETUP !! ---
# (เราจะตั้งค่าตัวแปรที่ใช้ร่วมกันก่อน)

print("\n" + "="*50)
print("=== Price Optimization (ACTIVATED) ===")
print("="*50)

# 1. ค้นหา SKUs ที่เราต้องการ
# (เราต้องแปลง SKU "ดิบ" ไปเป็น SKU "ที่เข้ารหัส")
target_skus_raw = ['1298', '2963', '4488']
try:
    target_skus_encoded = encoders['SKU'].transform(target_skus_raw)
    print(f"Optimizing for Raw SKUs: {target_skus_raw} (Encoded: {target_skus_encoded})")
except ValueError as e:
    print(f"Error: One of the target SKUs {target_skus_raw} was not found in the training data.")
    print("Skipping Price Optimization.")
    target_skus_encoded = None

if target_skus_encoded is not None:
    PRODUCT_COSTS = np.array([300, 400, 600]) # (ปรับต้นทุนตามจริง)
    NUM_PRODUCTS = len(target_skus_encoded)

    # 2. ดึง "ข้อมูลฐาน" (สัปดาห์ล่าสุด)
    base_features_unscaled = df_test[features].iloc[-1].values
    f_map = {name: idx for idx, name in enumerate(features)} # (Map ชื่อ feature ไปยัง index)

    # 3. Price Bounds
    price_bounds = [
        (400, 800),
        (500, 1000),
        (700, 1500)
    ]

    # 4. สร้าง Objective Function (ที่รับ model เป็น argument)
    def profit_objective_function(prices, model_to_use):
        """
        ฟังก์ชันคำนวณกำไร (สำหรับโมเดล Flat/Weekly)
        """
        model_inputs = []
        
        for i, new_price in enumerate(prices):
            future_features_unscaled = base_features_unscaled.copy()
            today_price = future_features_unscaled[f_map['Avg_Price']]
            
            future_features_unscaled[f_map['SKU']] = target_skus_encoded[i]
            future_features_unscaled[f_map['Avg_Price']] = new_price
            future_features_unscaled[f_map['Max_Price']] = new_price
            future_features_unscaled[f_map['Min_Price']] = new_price
            future_features_unscaled[f_map['Price_lag_1']] = today_price
            future_features_unscaled[f_map['price_change_pct']] = (new_price - today_price) / (today_price + 1e-6)
            future_features_unscaled[f_map['Has_Promotion']] = 0 
            
            model_inputs.append(future_features_unscaled)

        model_inputs_scaled = scaler_X.transform(np.array(model_inputs))
        
        predictions_raw = model_to_use.predict(model_inputs_scaled)
        predictions_raw = np.maximum(0, predictions_raw)
        
        total_profit = np.sum((prices - PRODUCT_COSTS) * predictions_raw)
        return -total_profit  # ติดลบเพราะ PSO minimize

    # --- !! 12. RUN OPTIMIZER FOR ALL 3 MODELS !! ---
    
    for model_name, data in results.items():
        
        print(f"\n--- Optimizing Prices using: {model_name} (R²: {data['R²']:.4f}) ---")
        
        current_model = data['model_obj']
        
        # (สร้าง lambda function เพื่อส่ง model ที่ถูกต้องเข้าไป)
        objective_wrapper = functools.partial(profit_objective_function, model_to_use=current_model)
        
        optimizer = ParticleSwarmOptimizer(
            objective_function=objective_wrapper,
            bounds=price_bounds,
            num_particles=50,
            max_iter=100
        )

        optimal_prices, max_profit = optimizer.optimize()

        print(f"\n{'='*50}")
        print(f"OPTIMIZATION RESULTS (For {model_name}):")
        print(f"{'='*50}")
        print(f"  Optimal Prices: {np.round(optimal_prices, 2)}")
        print(f"  Maximum Profit: ฿{-max_profit:,.2f}")
        print(f"{'='*50}")