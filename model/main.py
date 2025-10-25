# main.py (WEEKLY VERSION - COMPARE 3 MODELS)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- !! 1. IMPORT MODELS FOR COMPARISON !! ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor # (Simple Neural Network)

from aggregate_weekly import aggregate_data

# (ส่วนที่ 3, 4, 5, 6, 7 เหมือนเดิมเป๊ะ... ผมย่อไว้ให้อ่านง่าย)

# --- 3. LOAD WEEKLY DATA ---
DATA_FILE = r'E:\model\model\Amazon Sale Report.csv'
print("\n=== Loading and Aggregating Weekly Data ===")
df = aggregate_data(DATA_FILE)

# --- 4. ENCODE CATEGORICAL FEATURES ---
print("\nEncoding categorical features (SKU, Category, Size)...")
encoders = {}
for col in ['SKU', 'Category', 'Size']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
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
    'Has_Promotion' 
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
X_val_raw = df_val[features].values
y_val_target = df_val[target].values
X_test_raw = df_test[features].values
y_test_raw = df_test[target].values

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_val_scaled = scaler_X.transform(X_val_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

print(f"X_train shape (flat): {X_train_scaled.shape}")
print(f"X_test shape (flat): {X_test_scaled.shape}")

# --- !! 9. (REMOVED) สร้าง Sequences ---
# (เราไม่ต้องการ Sequences อีกต่อไป)

# --- !! 10. (REPLACED) Build, Train, and Compare Models ---
print("\n=== Training and Comparing Models ===")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10, # (ป้องกัน Overfit)
        random_state=42,
        n_jobs=-1
    ),
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 32), # (โครงสร้าง 2 ชั้น คล้าย LSTM เดิม)
        activation='relu',
        random_state=42,
        max_iter=500,
        early_stopping=True # (เปิด early stopping กัน Overfit)
    )
}

results = {}
best_model = None
best_r2 = -np.inf

print(f"\nTest Set Performance (Actual Mean: {y_test_raw.mean():.2f} units)")
print("-" * 50)

for name, model in models.items():
    print(f"Training: {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train_target)
    
    # Predict
    predictions_raw = model.predict(X_test_scaled)
    
    # Evaluate
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    r2 = r2_score(y_test_raw, predictions_raw)
    
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'model_obj': model, 'preds': predictions_raw}
    
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.2f} units")
    print(f"  RMSE: {rmse:.2f} units")
    print("-" * 50)
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = name

print(f"\n🏆 Best Model: {best_model} (Based on R²) 🏆")

# --- !! 11. (MODIFIED) Visualization (Plotting the BEST model) ---

best_model_preds = results[best_model]['preds']

plt.figure(figsize=(10, 5))

# Plot 1: Actual vs Predicted (Best Model)
plt.subplot(1, 2, 1)
plt.scatter(y_test_raw, best_model_preds, alpha=0.5)
plt.plot([y_test_raw.min(), y_test_raw.max()],
         [y_test_raw.min(), y_test_raw.max()], 'r--', lw=2)
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title(f'Actual vs Predicted ({best_model})\n(R²={results[best_model]["R²"]:.4f})')
plt.grid(True)

# Plot 2: Residuals (Best Model)
plt.subplot(1, 2, 2)
residuals = y_test_raw - best_model_preds
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title(f'Residual Distribution ({best_model})')
plt.grid(True)

plt.tight_layout()
plt.savefig(f'model_evaluation_weekly_COMPARISON.png', dpi=150)
print(f"\nVisualization for best model ({best_model}) saved to: model_evaluation_weekly_COMPARISON.png")

# --- 13. Price Optimization (SKIPPED) ---
print("\n=== Price Optimization (SKIPPED) ===")
print("Skipping PSO section for now.")