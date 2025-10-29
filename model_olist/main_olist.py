# main_olist_with_simple_evaluation.py
# เพิ่มเฉพาะส่วน Evaluation ที่จำเป็นในโค้ดเดิม

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import holidays
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import time

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
MIN_SALES_THRESHOLD = 50
TOP_N_PRODUCTS = 500
NUM_PRODUCTS_TO_OPTIMIZE = 3
PRODUCT_COSTS = [10, 15, 20]

IMPORTANT_LAGS = [1, 4]
IMPORTANT_ROLLS = [4]
PRICE_GRID_POINTS = 20

# =============================================================================
# SIMPLE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):
    """ฟังก์ชันประเมินโมเดลแบบง่าย"""
    
    # metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # MAPE
    mask = y_test != 0
    test_mape = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100
    
    # แสดงผล
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"{'Metric':<20} {'Train':>15} {'Test':>15} {'Diff':>12}")
    print("-"*70)
    print(f"{'MAE':<20} {train_mae:>15.4f} {test_mae:>15.4f} {test_mae-train_mae:>12.4f}")
    print(f"{'RMSE':<20} {train_rmse:>15.4f} {test_rmse:>15.4f} {test_rmse-train_rmse:>12.4f}")
    print(f"{'R² Score':<20} {train_r2:>15.4f} {test_r2:>15.4f} {test_r2-train_r2:>12.4f}")
    print(f"{'MAPE (%)':<20} {'-':>15} {test_mape:>15.2f} {'-':>12}")
    print("="*70)
    
    # เปรียบเทียบผล
    print("\n📊 INTERPRETATION:")
    if test_r2 > 0.8:
        print("✓ Excellent model (R² > 0.8)")
    elif test_r2 > 0.7:
        print("✓ Good model (R² > 0.7)")
    elif test_r2 > 0.6:
        print("⚠ Fair model (R² > 0.6)")
    else:
        print("✗ Poor model (R² < 0.6)")
    
    # เช็ค overfitting
    r2_diff = train_r2 - test_r2
    if r2_diff < 0.1:
        print("✓ No overfitting (Train-Test R² < 0.1)")
    elif r2_diff < 0.2:
        print("⚠ Slight overfitting (Train-Test R² < 0.2)")
    else:
        print("✗ Significant overfitting (Train-Test R² > 0.2)")
    
    if test_mape < 20:
        print(f"✓ Good accuracy (MAPE = {test_mape:.1f}%)")
    else:
        print(f"⚠ Consider improving (MAPE = {test_mape:.1f}%)")
    
    return {
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'test_mape': test_mape
    }


def plot_simple_evaluation(y_test, y_pred_test, save_path=None):
    """สร้าง 2 plots พื้นฐาน: Predicted vs Actual และ Residuals"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1: Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred_test, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Quantity Sold', fontsize=12)
    ax1.set_ylabel('Predicted Quantity Sold', fontsize=12)
    ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    r2 = r2_score(y_test, y_pred_test)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2: Residuals
    ax2 = axes[1]
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Quantity Sold', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def cross_validate_simple(model, X, y, cv=5):
    """Cross-validation แบบง่าย"""
    print("\n=== Cross-Validation (5-Fold) ===")
    
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    
    print(f"MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std():.4f})")
    print(f"R²:  {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
    
    if r2_scores.std() < 0.05:
        print("✓ Model is stable (low variance)")
    else:
        print("⚠ Model has high variance across folds")


# =============================================================================
# DATA LOADING & PROCESSING
# =============================================================================
print("\n=== Loading Olist Data ===")
try:
    df_orders = pd.read_csv('olist_orders_dataset.csv')
    df_items = pd.read_csv('olist_order_items_dataset.csv')
    df_products = pd.read_csv('olist_products_dataset.csv')
    df_trans = pd.read_csv('product_category_name_translation.csv')
    df_sellers = pd.read_csv('olist_sellers_dataset.csv')
    print("✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    exit()

print("\n=== Processing Data ===")
start_time = time.time()

df_orders = df_orders[df_orders['order_status'] == 'delivered'].copy()
df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])

df = pd.merge(df_orders, df_items, on='order_id', how='inner')
df = pd.merge(df, df_products, on='product_id', how='left')
df = pd.merge(df, df_trans, on='product_category_name', how='left')
df = pd.merge(df, df_sellers[['seller_id', 'seller_state']], on='seller_id', how='left')

df['seller_state'] = df['seller_state'].fillna('Unknown')
le_state = LabelEncoder()
df['seller_state_encoded'] = le_state.fit_transform(df['seller_state'])

df['quantity'] = 1
df_agg = df[[
    'order_purchase_timestamp', 'product_id', 'product_category_name_english',
    'seller_state_encoded', 'price', 'quantity',
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'
]].copy()

df_agg.dropna(subset=['product_category_name_english'], inplace=True)
print(f"✓ Initial records: {len(df_agg):,}")

print("\n=== Aggregating to Weekly ===")
df_agg['Date'] = df_agg['order_purchase_timestamp']
df_agg = df_agg.set_index('Date')

weekly_data = df_agg.groupby([
    'product_id', 'product_category_name_english', 'seller_state_encoded',
    pd.Grouper(freq='W-MON')
]).agg(
    QuantitySold=('quantity', 'count'),
    AverageSellingPrice=('price', 'mean'),
    Weight_g_Mean=('product_weight_g', 'mean'),
    Length_cm_Mean=('product_length_cm', 'mean'),
    Height_cm_Mean=('product_height_cm', 'mean'),
    Width_cm_Mean=('product_width_cm', 'mean')
).reset_index()

print(f"✓ Weekly records (raw): {len(weekly_data):,}")

print("\n=== Filtering Popular Products ===")
product_sales = weekly_data.groupby('product_id')['QuantitySold'].sum().sort_values(ascending=False)

if MIN_SALES_THRESHOLD:
    popular_products = product_sales[product_sales >= MIN_SALES_THRESHOLD].index
else:
    popular_products = product_sales.head(TOP_N_PRODUCTS).index

print(f"✓ Popular products: {len(popular_products):,}")

weekly_data = weekly_data[weekly_data['product_id'].isin(popular_products)].copy()
print(f"✓ Weekly records (filtered): {len(weekly_data):,}")

df = weekly_data.copy()

df['AverageSellingPrice'] = df.groupby('product_id')['AverageSellingPrice'].ffill().bfill()
df['AverageSellingPrice'] = df['AverageSellingPrice'].fillna(df['AverageSellingPrice'].median())

prod_features = ['Weight_g_Mean', 'Length_cm_Mean', 'Height_cm_Mean', 'Width_cm_Mean']
for col in prod_features:
    df[col] = df.groupby('product_id')[col].ffill().bfill()
    df[col] = df[col].fillna(0)

print(f"✓ Processing time: {time.time() - start_time:.1f}s")

print("\n=== Feature Engineering ===")
df = df.sort_values(by=['product_id', 'Date'])

le_cat = LabelEncoder()
df['category_encoded'] = le_cat.fit_transform(df['product_category_name_english'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Quarter'] = df['Date'].dt.quarter

print("  - Creating holiday features...")
br_holidays = holidays.Brazil(years=range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1))
df['IsHoliday'] = df['Date'].apply(lambda d: 1 if d in br_holidays else 0)

print("  - Creating lag & rolling features...")

def create_lag_features(group, col, lags=IMPORTANT_LAGS):
    for lag in lags:
        group[f'{col}_Lag_{lag}'] = group[col].shift(lag)
    return group

def create_rolling_features(group, col, windows=IMPORTANT_ROLLS):
    for window in windows:
        group[f'{col}_Roll_Mean_{window}'] = group[col].rolling(window=window, min_periods=1).mean()
    return group

df = df.groupby('product_id', group_keys=False).apply(
    lambda g: create_rolling_features(create_lag_features(g, 'AverageSellingPrice'), 'AverageSellingPrice')
)
df = df.groupby('product_id', group_keys=False).apply(
    lambda g: create_rolling_features(create_lag_features(g, 'QuantitySold'), 'QuantitySold')
)

df['Price_Diff_Lag_1'] = df.groupby('product_id')['AverageSellingPrice'].diff()
df['Qty_Diff_Lag_1'] = df.groupby('product_id')['QuantitySold'].diff()

df = df.fillna(0)

print(f"✓ Total features: {len(df.columns)}")
print(f"✓ Total records: {len(df):,}")

# =============================================================================
# PREPARE TRAIN/TEST
# =============================================================================
print("\n=== Preparing Train/Test Split ===")

feature_cols = [
    'category_encoded', 'seller_state_encoded',
    'AverageSellingPrice', 'Weight_g_Mean', 'Length_cm_Mean', 'Height_cm_Mean', 'Width_cm_Mean',
    'Year', 'Month', 'Week', 'Quarter', 'IsWeekend', 'IsHoliday'
]

for col in ['AverageSellingPrice', 'QuantitySold']:
    for lag in IMPORTANT_LAGS:
        feature_cols.append(f'{col}_Lag_{lag}')
    for window in IMPORTANT_ROLLS:
        feature_cols.append(f'{col}_Roll_Mean_{window}')

feature_cols.extend(['Price_Diff_Lag_1', 'Qty_Diff_Lag_1'])

X = df[feature_cols].values
y = df['QuantitySold'].values

split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✓ Train: {len(X_train):,} samples")
print(f"✓ Test:  {len(X_test):,} samples")

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# =============================================================================
# TRAIN MODEL
# =============================================================================
print("\n=== Training LightGBM Model ===")
start_time = time.time()

model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print(f"✓ Training time: {train_time:.2f}s")

# =============================================================================
# MODEL EVALUATION
# =============================================================================

# 1. แสดง metrics 
metrics = evaluate_model(y_train, y_pred_train, y_test, y_pred_test)

# 2. สร้าง plots
print("\n=== Creating Evaluation Plots ===")
plot_simple_evaluation(y_test, y_pred_test, 
                       save_path='/mnt/user-data/outputs/model_evaluation.png')

# 3. Cross-validation
print("\nRunning cross-validation on sample...")
sample_size = min(10000, len(X_train_scaled))
sample_idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)
cross_validate_simple(model, X_train_scaled[sample_idx], y_train[sample_idx], cv=3)

# =============================================================================
# PRICE OPTIMIZATION 
# =============================================================================
print("\n=== Price Optimization (Grid Search) ===")

df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

target_products = df_train.groupby('product_id')['QuantitySold'].sum().nlargest(NUM_PRODUCTS_TO_OPTIMIZE).index.values
print(f"✓ Target products: {target_products}")

base_data = df_train.groupby('product_id').last()

price_stats = df_train[df_train['product_id'].isin(target_products)].groupby('product_id')['AverageSellingPrice'].agg(['mean', 'std'])
print(f"\n✓ Price Statistics:")
print(price_stats)


def optimize_price_grid(model, prod_id, price_bounds, cost, scaler, feature_cols):
    """หาราคาที่ดีที่สุดด้วย Grid Search"""
    prices = np.linspace(price_bounds[0], price_bounds[1], PRICE_GRID_POINTS)
    base_features = base_data.loc[prod_id][feature_cols].values
    
    best_price = None
    best_profit = -np.inf
    
    for price in prices:
        features_unscaled = base_features.copy()
        
        price_idx = feature_cols.index('AverageSellingPrice')
        features_unscaled[price_idx] = price
        
        if 'AverageSellingPrice_Lag_1' in feature_cols:
            lag_idx = feature_cols.index('AverageSellingPrice_Lag_1')
            features_unscaled[lag_idx] = base_features[price_idx]
        
        features_scaled = scaler.transform(features_unscaled.reshape(1, -1))
        predicted_qty = model.predict(features_scaled)[0]
        predicted_qty = max(0, round(predicted_qty))
        
        profit = (price - cost) * predicted_qty
        
        if profit > best_profit:
            best_profit = profit
            best_price = price
    
    return best_price, best_profit


print("\n" + "="*60)
optimization_results = {}

for i, prod_id in enumerate(target_products):
    print(f"\nOptimizing Product {i+1}/{len(target_products)}: {prod_id}")
    
    mean_price = price_stats.loc[prod_id, 'mean']
    price_bounds = (mean_price * 0.7, mean_price * 1.3)
    cost = PRODUCT_COSTS[i]
    
    print(f"  Current avg price: ฿{mean_price:.2f}")
    print(f"  Price range: ฿{price_bounds[0]:.2f} - ฿{price_bounds[1]:.2f}")
    print(f"  Cost: ฿{cost:.2f}")
    
    start_time = time.time()
    optimal_price, max_profit = optimize_price_grid(
        model, prod_id, price_bounds, cost, scaler_X, feature_cols
    )
    opt_time = time.time() - start_time
    
    optimization_results[prod_id] = {
        'optimal_price': optimal_price,
        'max_profit': max_profit,
        'current_price': mean_price,
        'time': opt_time
    }
    
    print(f"  ✓ Optimal price: ฿{optimal_price:.2f}")
    print(f"  ✓ Expected profit: ฿{max_profit:,.2f}")
    print(f"  ✓ Optimization time: {opt_time:.3f}s")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("OPTIMIZATION SUMMARY")
print("="*60)

total_profit = 0
for prod_id, result in optimization_results.items():
    print(f"\nProduct {prod_id}:")
    print(f"  Current Price: ฿{result['current_price']:.2f}")
    print(f"  Optimal Price: ฿{result['optimal_price']:.2f} ({result['optimal_price']/result['current_price']*100-100:+.1f}%)")
    print(f"  Expected Profit: ฿{result['max_profit']:,.2f}")
    total_profit += result['max_profit']

print(f"\n{'='*60}")
print(f"TOTAL EXPECTED PROFIT: ฿{total_profit:,.2f}")
print(f"{'='*60}")

# main_olist_with_simple_evaluation.py
# เพิ่มเฉพาะส่วน Evaluation ที่จำเป็นในโค้ดเดิม

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import holidays
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import time

from price_optimizer import ParticleSwarmOptimizer
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
MIN_SALES_THRESHOLD = 50
TOP_N_PRODUCTS = 500
NUM_PRODUCTS_TO_OPTIMIZE = 3
PRODUCT_COSTS = [10, 15, 20]

IMPORTANT_LAGS = [1, 4]
IMPORTANT_ROLLS = [4]
PRICE_GRID_POINTS = 20

# =============================================================================
# เพิ่ม: SIMPLE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):
    """ฟังก์ชันประเมินโมเดลแบบง่าย"""
    
    # คำนวณ metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # คำนวณ MAPE
    mask = y_test != 0
    test_mape = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100
    
    # แสดงผล
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"{'Metric':<20} {'Train':>15} {'Test':>15} {'Diff':>12}")
    print("-"*70)
    print(f"{'MAE':<20} {train_mae:>15.4f} {test_mae:>15.4f} {test_mae-train_mae:>12.4f}")
    print(f"{'RMSE':<20} {train_rmse:>15.4f} {test_rmse:>15.4f} {test_rmse-train_rmse:>12.4f}")
    print(f"{'R² Score':<20} {train_r2:>15.4f} {test_r2:>15.4f} {test_r2-train_r2:>12.4f}")
    print(f"{'MAPE (%)':<20} {'-':>15} {test_mape:>15.2f} {'-':>12}")
    print("="*70)
    
    # ตีความผล
    print("\n📊 INTERPRETATION:")
    if test_r2 > 0.8:
        print("✓ Excellent model (R² > 0.8)")
    elif test_r2 > 0.7:
        print("✓ Good model (R² > 0.7)")
    elif test_r2 > 0.6:
        print("⚠ Fair model (R² > 0.6)")
    else:
        print("✗ Poor model (R² < 0.6)")
    
    # เช็ค overfitting
    r2_diff = train_r2 - test_r2
    if r2_diff < 0.1:
        print("✓ No overfitting (Train-Test R² < 0.1)")
    elif r2_diff < 0.2:
        print("⚠ Slight overfitting (Train-Test R² < 0.2)")
    else:
        print("✗ Significant overfitting (Train-Test R² > 0.2)")
    
    if test_mape < 20:
        print(f"✓ Good accuracy (MAPE = {test_mape:.1f}%)")
    else:
        print(f"⚠ Consider improving (MAPE = {test_mape:.1f}%)")
    
    return {
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'test_mape': test_mape
    }


def plot_simple_evaluation(y_test, y_pred_test, save_path=None):
    """สร้าง 2 plots พื้นฐาน: Predicted vs Actual และ Residuals"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred_test, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Quantity Sold', fontsize=12)
    ax1.set_ylabel('Predicted Quantity Sold', fontsize=12)
    ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    r2 = r2_score(y_test, y_pred_test)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Residuals
    ax2 = axes[1]
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Quantity Sold', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def cross_validate_simple(model, X, y, cv=5):
    """Cross-validation แบบง่าย"""
    print("\n=== Cross-Validation (5-Fold) ===")
    
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    
    print(f"MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std():.4f})")
    print(f"R²:  {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
    
    if r2_scores.std() < 0.05:
        print("✓ Model is stable (low variance)")
    else:
        print("⚠ Model has high variance across folds")


# =============================================================================
# 1-6. DATA LOADING & PROCESSING (เหมือนเดิม)
# =============================================================================
print("\n=== Loading Olist Data ===")
try:
    df_orders = pd.read_csv('olist_orders_dataset.csv')
    df_items = pd.read_csv('olist_order_items_dataset.csv')
    df_products = pd.read_csv('olist_products_dataset.csv')
    df_trans = pd.read_csv('product_category_name_translation.csv')
    df_sellers = pd.read_csv('olist_sellers_dataset.csv')
    print("✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    exit()

print("\n=== Processing Data ===")
start_time = time.time()

df_orders = df_orders[df_orders['order_status'] == 'delivered'].copy()
df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])

df = pd.merge(df_orders, df_items, on='order_id', how='inner')
df = pd.merge(df, df_products, on='product_id', how='left')
df = pd.merge(df, df_trans, on='product_category_name', how='left')
df = pd.merge(df, df_sellers[['seller_id', 'seller_state']], on='seller_id', how='left')

df['seller_state'] = df['seller_state'].fillna('Unknown')
le_state = LabelEncoder()
df['seller_state_encoded'] = le_state.fit_transform(df['seller_state'])

df['quantity'] = 1
df_agg = df[[
    'order_purchase_timestamp', 'product_id', 'product_category_name_english',
    'seller_state_encoded', 'price', 'quantity',
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'
]].copy()

df_agg.dropna(subset=['product_category_name_english'], inplace=True)
print(f"✓ Initial records: {len(df_agg):,}")

print("\n=== Aggregating to Weekly ===")
df_agg['Date'] = df_agg['order_purchase_timestamp']
df_agg = df_agg.set_index('Date')

weekly_data = df_agg.groupby([
    'product_id', 'product_category_name_english', 'seller_state_encoded',
    pd.Grouper(freq='W-MON')
]).agg(
    QuantitySold=('quantity', 'count'),
    AverageSellingPrice=('price', 'mean'),
    Weight_g_Mean=('product_weight_g', 'mean'),
    Length_cm_Mean=('product_length_cm', 'mean'),
    Height_cm_Mean=('product_height_cm', 'mean'),
    Width_cm_Mean=('product_width_cm', 'mean')
).reset_index()

print(f"✓ Weekly records (raw): {len(weekly_data):,}")

print("\n=== Filtering Popular Products ===")
product_sales = weekly_data.groupby('product_id')['QuantitySold'].sum().sort_values(ascending=False)

if MIN_SALES_THRESHOLD:
    popular_products = product_sales[product_sales >= MIN_SALES_THRESHOLD].index
else:
    popular_products = product_sales.head(TOP_N_PRODUCTS).index

print(f"✓ Popular products: {len(popular_products):,}")

weekly_data = weekly_data[weekly_data['product_id'].isin(popular_products)].copy()
print(f"✓ Weekly records (filtered): {len(weekly_data):,}")

df = weekly_data.copy()

df['AverageSellingPrice'] = df.groupby('product_id')['AverageSellingPrice'].ffill().bfill()
df['AverageSellingPrice'] = df['AverageSellingPrice'].fillna(df['AverageSellingPrice'].median())

prod_features = ['Weight_g_Mean', 'Length_cm_Mean', 'Height_cm_Mean', 'Width_cm_Mean']
for col in prod_features:
    df[col] = df.groupby('product_id')[col].ffill().bfill()
    df[col] = df[col].fillna(0)

print(f"✓ Processing time: {time.time() - start_time:.1f}s")

print("\n=== Feature Engineering ===")
df = df.sort_values(by=['product_id', 'Date'])

le_cat = LabelEncoder()
df['category_encoded'] = le_cat.fit_transform(df['product_category_name_english'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Quarter'] = df['Date'].dt.quarter

print("  - Creating holiday features...")
br_holidays = holidays.Brazil(years=range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1))
df['IsHoliday'] = df['Date'].apply(lambda d: 1 if d in br_holidays else 0)

print("  - Creating lag & rolling features...")

def create_lag_features(group, col, lags=IMPORTANT_LAGS):
    for lag in lags:
        group[f'{col}_Lag_{lag}'] = group[col].shift(lag)
    return group

def create_rolling_features(group, col, windows=IMPORTANT_ROLLS):
    for window in windows:
        group[f'{col}_Roll_Mean_{window}'] = group[col].rolling(window=window, min_periods=1).mean()
    return group

df = df.groupby('product_id', group_keys=False).apply(
    lambda g: create_rolling_features(create_lag_features(g, 'AverageSellingPrice'), 'AverageSellingPrice')
)
df = df.groupby('product_id', group_keys=False).apply(
    lambda g: create_rolling_features(create_lag_features(g, 'QuantitySold'), 'QuantitySold')
)

df['Price_Diff_Lag_1'] = df.groupby('product_id')['AverageSellingPrice'].diff()
df['Qty_Diff_Lag_1'] = df.groupby('product_id')['QuantitySold'].diff()

df = df.fillna(0)

print(f"✓ Total features: {len(df.columns)}")
print(f"✓ Total records: {len(df):,}")

# =============================================================================
# PREPARE TRAIN/TEST
# =============================================================================
print("\n=== Preparing Train/Test Split ===")

feature_cols = [
    'category_encoded', 'seller_state_encoded',
    'AverageSellingPrice', 'Weight_g_Mean', 'Length_cm_Mean', 'Height_cm_Mean', 'Width_cm_Mean',
    'Year', 'Month', 'Week', 'Quarter', 'IsWeekend', 'IsHoliday'
]

for col in ['AverageSellingPrice', 'QuantitySold']:
    for lag in IMPORTANT_LAGS:
        feature_cols.append(f'{col}_Lag_{lag}')
    for window in IMPORTANT_ROLLS:
        feature_cols.append(f'{col}_Roll_Mean_{window}')

feature_cols.extend(['Price_Diff_Lag_1', 'Qty_Diff_Lag_1'])

X = df[feature_cols].values
y = df['QuantitySold'].values

split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✓ Train: {len(X_train):,} samples")
print(f"✓ Test:  {len(X_test):,} samples")

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# =============================================================================
# TRAIN MODEL
# =============================================================================
print("\n=== Training LightGBM Model ===")
start_time = time.time()

model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print(f"✓ Training time: {train_time:.2f}s")

# =============================================================================
# เพิ่ม: MODEL EVALUATION
# =============================================================================

# 1. แสดง metrics แบบละเอียด
metrics = evaluate_model(y_train, y_pred_train, y_test, y_pred_test)

# 2. สร้าง plots
print("\n=== Creating Evaluation Plots ===")
plot_simple_evaluation(y_test, y_pred_test, 
                       save_path='/mnt/user-data/outputs/model_evaluation.png')

# 3. Cross-validation
print("\nRunning cross-validation on sample...")
sample_size = min(10000, len(X_train_scaled))
sample_idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)
cross_validate_simple(model, X_train_scaled[sample_idx], y_train[sample_idx], cv=3)

# =============================================================================
# PRICE OPTIMIZATION (เหมือนเดิม)
# =============================================================================
print("\n=== Price Optimization (Grid Search) ===")

df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

target_products = df_train.groupby('product_id')['QuantitySold'].sum().nlargest(NUM_PRODUCTS_TO_OPTIMIZE).index.values
print(f"✓ Target products: {target_products}")

base_data = df_train.groupby('product_id').last()

price_stats = df_train[df_train['product_id'].isin(target_products)].groupby('product_id')['AverageSellingPrice'].agg(['mean', 'std'])
print(f"\n✓ Price Statistics:")
print(price_stats)



def optimize_price_grid(model, prod_id, price_bounds, cost, scaler, feature_cols):
    """หาราคาที่ดีที่สุดด้วย Particle Swarm Optimization (PSO)"""
    
    base_features = base_data.loc[prod_id][feature_cols].values
    price_idx = feature_cols.index('AverageSellingPrice')
    
    # สร้าง objective function สำหรับ PSO (minimize negative profit)
    def objective_function(params):
        price = params[0]
        
        features_unscaled = base_features.copy()
        features_unscaled[price_idx] = price
        
        # อัพเดท lag feature ถ้ามี
        if 'AverageSellingPrice_Lag_1' in feature_cols:
            lag_idx = feature_cols.index('AverageSellingPrice_Lag_1')
            features_unscaled[lag_idx] = base_features[price_idx]
        
        # Predict demand
        features_scaled = scaler.transform(features_unscaled.reshape(1, -1))
        predicted_qty = model.predict(features_scaled)[0]
        predicted_qty = max(0, round(predicted_qty))
        
        # Calculate profit (negative for minimization)
        profit = (price - cost) * predicted_qty
        return -profit  # PSO minimizes, so we negate
    
    # ใช้ PSO แทน Grid Search
    pso = ParticleSwarmOptimizer(
        objective_function=objective_function,
        bounds=[(price_bounds[0], price_bounds[1])],  # Single dimension (price)
        num_particles=30,
        max_iter=50,
        verbose=False  # ปิด verbose เพื่อไม่ให้มี output เยอะเกินไป
    )
    
    best_params, best_value = pso.optimize()
    best_price = best_params[0]
    best_profit = -best_value  # Convert back to positive profit
    
    return best_price, best_profit



print("\n" + "="*60)
optimization_results = {}

for i, prod_id in enumerate(target_products):
    print(f"\nOptimizing Product {i+1}/{len(target_products)}: {prod_id}")
    
    mean_price = price_stats.loc[prod_id, 'mean']
    price_bounds = (mean_price * 0.7, mean_price * 1.3)
    cost = PRODUCT_COSTS[i]
    
    print(f"  Current avg price: ฿{mean_price:.2f}")
    print(f"  Price range: ฿{price_bounds[0]:.2f} - ฿{price_bounds[1]:.2f}")
    print(f"  Cost: ฿{cost:.2f}")
    
    start_time = time.time()
    optimal_price, max_profit = optimize_price_grid(
        model, prod_id, price_bounds, cost, scaler_X, feature_cols
    )
    opt_time = time.time() - start_time
    
    optimization_results[prod_id] = {
        'optimal_price': optimal_price,
        'max_profit': max_profit,
        'current_price': mean_price,
        'time': opt_time
    }
    
    print(f"  ✓ Optimal price: ฿{optimal_price:.2f}")
    print(f"  ✓ Expected profit: ฿{max_profit:,.2f}")
    print(f"  ✓ Optimization time: {opt_time:.3f}s")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("OPTIMIZATION SUMMARY")
print("="*60)

total_profit = 0
for prod_id, result in optimization_results.items():
    print(f"\nProduct {prod_id}:")
    print(f"  Current Price: ฿{result['current_price']:.2f}")
    print(f"  Optimal Price: ฿{result['optimal_price']:.2f} ({result['optimal_price']/result['current_price']*100-100:+.1f}%)")
    print(f"  Expected Profit: ฿{result['max_profit']:,.2f}")
    total_profit += result['max_profit']

print(f"\n{'='*60}")
print(f"TOTAL EXPECTED PROFIT: ฿{total_profit:,.2f}")
print(f"{'='*60}")

print("\n✓ Script finished successfully!")