# main_walmart.py (Compare 3 Models on Walmart Data)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import functools

# --- !! 1. IMPORT MODELS FOR COMPARISON !! ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# (เราไม่ต้องการ aggregate_weekly.py)

# --- !! 2. IMPORT PRICE OPTIMIZER FROM CURRENT FOLDER !! ---
# (ตรวจสอบว่า price_optimizer.py อยู่ในโฟลเดอร์เดียวกัน)
try:
    from price_optimizer import ParticleSwarmOptimizer
    PSO_ENABLED = True
except ImportError:
    print("Warning: price_optimizer.py not found. Price Optimization section will be skipped.")
    PSO_ENABLED = False


# --- !! 3. LOAD WALMART DATA !! ---
print("\n=== Loading Walmart Data ===")
try:
    # (โค้ดจะหาไฟล์ในโฟลเดอร์เดียวกัน)
    df_train = pd.read_csv('train.csv')
    df_features = pd.read_csv('features.csv')
    df_stores = pd.read_csv('stores.csv')
    print("Walmart CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not load Walmart CSV file - {e}")
    print("Make sure train.csv, features.csv, and stores.csv are in the same folder as this script.")
    exit()

# --- !! 4. MERGE DATA !! ---
print("\nMerging datasets...")
# แปลง Date เป็น datetime ก่อน Merge
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_features['Date'] = pd.to_datetime(df_features['Date'])

# Merge features -> stores -> train
df = pd.merge(df_features, df_stores, on='Store', how='left')
df = pd.merge(df_train, df, on=['Store', 'Date', 'IsHoliday'], how='left')

print(f"Merged DataFrame shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# --- !! 5. FEATURE ENGINEERING FOR WALMART !! ---
print("\n=== Creating Walmart Features ===")

# Handle Missing Values
print("Handling missing values...")
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
df[markdown_cols] = df[markdown_cols].fillna(0)
df['CPI'] = df['CPI'].fillna(df['CPI'].mean())
df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].mean())

# Time Features
print("Creating time features...")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(float)
df['Day'] = df['Date'].dt.day
df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
df['Week_Sin'] = np.sin(2 * np.pi * df['Week']/52)
df['Week_Cos'] = np.cos(2 * np.pi * df['Week']/52)

# Lag Features
print("Creating lag features...")
df = df.sort_values(by=['Store', 'Dept', 'Date'])
df['Weekly_Sales_Lag_1'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
df['Weekly_Sales_Lag_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)

# Rolling Mean Features
print("Creating rolling mean features...")
df['Weekly_Sales_Roll_Mean_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=4, min_periods=1).mean()

# Categorical Feature Encoding
print("\nEncoding categorical features (Type, IsHoliday)...")
encoders = {}
df['IsHoliday'] = df['IsHoliday'].astype(int)
le_type = LabelEncoder()
df['Type'] = le_type.fit_transform(df['Type'])
encoders['Type'] = le_type
# (Store และ Dept เป็นตัวเลขอยู่แล้ว)
encoders['Store'] = 'Numeric'
encoders['Dept'] = 'Numeric'

# Drop rows with NaNs
print(f"Shape before dropping NaNs: {df.shape}")
df = df.dropna(subset=['Weekly_Sales_Lag_1', 'Weekly_Sales_Lag_4', 'Weekly_Sales_Roll_Mean_4'])
print(f"Shape after dropping NaNs: {df.shape}")

# --- !! 6. WALMART FEATURE LIST !! ---
features = [
    'Store', 'Dept',
    'Year', 'Month', 'Week', 'Day',
    'Month_Sin', 'Month_Cos', 'Week_Sin', 'Week_Cos',
    'IsHoliday', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'Type', 'Size',
    'Weekly_Sales_Lag_1', 'Weekly_Sales_Lag_4', 'Weekly_Sales_Roll_Mean_4'
]
target = 'Weekly_Sales'
NUM_FEATURES = len(features)
print(f"\nTotal Walmart features: {NUM_FEATURES}")

# --- !! 7. Time-based Train/Validation Split !! ---
print("\n=== Time-based Train/Validation Split ===")
df = df.sort_values(by=['Date'])

total_weeks = df['Date'].nunique()
train_weeks = int(total_weeks * 0.8)
split_date = df['Date'].unique()[train_weeks]

print(f"Splitting data before date: {split_date}")

df_train = df[df['Date'] < split_date].copy()
df_val = df[df['Date'] >= split_date].copy() # ใช้ df_val เป็น Test Set ของเรา

print(f"Train period: {df_train['Date'].min()} to {df_train['Date'].max()}")
print(f"Validation period: {df_val['Date'].min()} to {df_val['Date'].max()}")

if len(df_train) == 0 or len(df_val) == 0:
    raise ValueError("Data splitting resulted in empty Train or Validation set.")

# --- 8. Fit Scaler ---
print("\n=== Fitting Scaler on Train Set ONLY ===")
X_train_raw = df_train[features].values
y_train_target = df_train[target].values
X_test_raw = df_val[features].values
y_test_raw = df_val[target].values

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

print(f"X_train shape (flat): {X_train_scaled.shape}")
print(f"X_test shape (flat): {X_test_scaled.shape}")

# --- 9. Build, Train, and Compare Models ---
print("\n=== Training and Comparing Models (Walmart) ===")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        verbose=1 # เพิ่ม verbose
    ),
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        random_state=42,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        verbose=True # เพิ่ม verbose
    )
}

results = {}
best_model_name = None
best_r2 = -np.inf

print(f"\nValidation Set Performance (Actual Mean: {y_test_raw.mean():.2f} units)")
print("-" * 50)

for name, model in models.items():
    print(f"Training: {name}...")
    model.fit(X_train_scaled, y_train_target)
    predictions_raw = model.predict(X_test_scaled)

    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))

    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'model_obj': model, 'preds': predictions_raw}

    print(f"Finished Training: {name}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.2f} units")
    print(f"  RMSE: {rmse:.2f} units")
    print("-" * 50)

    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name

print(f"\n🏆 Best Model on Validation Set: {best_model_name} (Based on R²) 🏆")

# --- 10. Visualization (Plotting the BEST model on Validation Set) ---
print("\nGenerating visualization...")
best_model_preds = results[best_model_name]['preds']
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test_raw, best_model_preds, alpha=0.1)
plt.plot([y_test_raw.min(), y_test_raw.max()],
         [y_test_raw.min(), y_test_raw.max()], 'r--', lw=2)
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.title(f'Actual vs Predicted ({best_model_name})\nValidation Set (R²={results[best_model_name]["R²"]:.4f})')
plt.grid(True)
plt.subplot(1, 2, 2)
residuals = y_test_raw - best_model_preds
# (ปรับ bins ตาม range ของ residuals)
res_min, res_max = residuals.min(), residuals.max()
bins = np.linspace(res_min, res_max, 100) # สร้าง bins ให้ละเอียดขึ้น
plt.hist(residuals, bins=bins, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title(f'Residual Distribution ({best_model_name})')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'model_evaluation_walmart_COMPARISON.png', dpi=150)
print(f"\nVisualization for best model ({best_model_name}) saved to: model_evaluation_walmart_COMPARISON.png")

# --- !! 11. Price Optimization (SKIPPED) !! ---
if PSO_ENABLED:
    print("\n" + "="*50)
    print("=== Price Optimization (SKIPPED - Requires Redesign) ===")
    print("="*50)
    print("Skipping PSO for Walmart data. Objective function needs Store/Dept logic and price features.")
    # (โค้ด PSO เดิมถูกลบออกชั่วคราว)
else:
    print("\nPrice Optimization skipped because price_optimizer.py was not found.")

print("\nScript finished.")