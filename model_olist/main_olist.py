import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import holidays
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import time, os

# XGBoost
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception as _e:
    print("⚠ XGBoost not available:", _e)
    _HAS_XGB = False

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# =============================================================================
# CONFIGURATION Fashion Products
# =============================================================================
MIN_SALES_THRESHOLD = 10  
TOP_N_PRODUCTS = 500
NUM_PRODUCTS_TO_OPTIMIZE = 5 
PRODUCT_COSTS = [10, 15, 20, 12, 18] 

def is_fashion_category(category_name):
    """ตรวจสอบว่าหมวดหมู่เป็น fashion หรือไม่ (เฉพาะที่ขึ้นต้นด้วย fashion_)"""
    if pd.isna(category_name):
        return False
    category_lower = str(category_name).lower()
    return category_lower.startswith('fashion_')

IMPORTANT_LAGS = [1, 4]
IMPORTANT_ROLLS = [4]
PRICE_GRID_POINTS = 20

# =============================================================================
# SIMPLE EVALUATION FUNCTIONS
# =============================================================================
def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    mask = y_test != 0
    test_mape = np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100 if mask.any() else np.nan

    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"{'Metric':<20} {'Train':>15} {'Test':>15} {'Diff':>12}")
    print("-"*70)
    print(f"{'MAE':<20} {train_mae:>15.4f} {test_mae:>15.4f} {test_mae-train_mae:>12.4f}")
    print(f"{'RMSE':<20} {train_rmse:>15.4f} {test_rmse:>15.4f} {test_rmse-train_rmse:>12.4f}")
    print(f"{'R² Score':<20} {train_r2:>15.4f} {test_r2:>15.4f} {test_r2-train_r2:>12.4f}")
    mape_txt = f"{test_mape:>.2f}" if pd.notnull(test_mape) else "NA"
    print(f"{'MAPE (%)':<20} {'-':>15} {mape_txt:>15} {'-':>12}")
    print("="*70)

    print("\n📊 INTERPRETATION:")
    if test_r2 > 0.8:
        print("✓ Excellent model (R² > 0.8)")
    elif test_r2 > 0.7:
        print("✓ Good model (R² > 0.7)")
    elif test_r2 > 0.6:
        print("⚠ Fair model (R² > 0.6)")
    else:
        print("✗ Poor model (R² < 0.6)")

    r2_diff = train_r2 - test_r2
    if r2_diff < 0.1:
        print("✓ No overfitting (Train-Test R² < 0.1)")
    elif r2_diff < 0.2:
        print("⚠ Slight overfitting (Train-Test R² < 0.2)")
    else:
        print("✗ Significant overfitting (Train-Test R² > 0.2)")

    if pd.notnull(test_mape):
        if test_mape < 20:
            print(f"✓ Good accuracy (MAPE = {test_mape:.1f}%)")
        else:
            print(f"⚠ Consider improving (MAPE = {test_mape:.1f}%)")

    return {
        'train_mae': float(train_mae), 'test_mae': float(test_mae),
        'train_rmse': float(train_rmse), 'test_rmse': float(test_rmse),
        'train_r2': float(train_r2), 'test_r2': float(test_r2),
        'test_mape': float(test_mape) if pd.notnull(test_mape) else np.nan
    }

def plot_simple_evaluation(y_test, y_pred_test, model_name="Model", save_path=None):
    """Plots Predicted vs Actual and Residuals for a given model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Evaluation Plots for: {model_name}', fontsize=16, fontweight='bold')

    # กราฟที่ 1: Predicted vs Actual
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

    # กราฟที่ 2: Residual Plot
    ax2 = axes[1]
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Quantity Sold', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    plt.show()

def cross_validate_simple(model, X, y, cv=5):
    print("\n=== Cross-Validation (5-Fold) ===")
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    print(f"MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std():.4f})")
    print(f"R²:  {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
    print("✓ Model is stable (low variance)" if r2_scores.std() < 0.05 else "⚠ Model has high variance across folds")

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
    raise SystemExit

print("\n=== Processing Data ===")
start_time = time.time()

df_orders = df_orders[df_orders['order_status'] == 'delivered'].copy()
df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])

df = pd.merge(df_orders, df_items, on='order_id', how='inner')
df = pd.merge(df, df_products, on='product_id', how='left')
df = pd.merge(df, df_trans, on='product_category_name', how='left')
df = pd.merge(df, df_sellers[['seller_id', 'seller_state']], on='seller_id', how='left')

# ===== กรองเฉพาะ Fashion Categories (fashion_* only) =====
print("\n=== Filtering Fashion Products Only (fashion_*) ===")
print(f"Records before filtering: {len(df):,}")

df = df[df['product_category_name_english'].apply(is_fashion_category)].copy()
print(f"✓ Records after fashion filter: {len(df):,}")
print(f"✓ Fashion categories found: {df['product_category_name_english'].nunique()}")
print(f"✓ Fashion products found: {df['product_id'].nunique()}")
print("\nFashion Categories (fashion_* only):")
print(df['product_category_name_english'].value_counts())

# ===== ดำเนินการต่อ =====
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
print(f"✓ Records after cleanup: {len(df_agg):,}")

print("\n=== Aggregating to Weekly (Product-Level) ===")
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

print("\n=== Filtering Popular Fashion Products ===")
product_sales = weekly_data.groupby('product_id')['QuantitySold'].sum().sort_values(ascending=False)

# กรองตามยอดขายขั้นต่ำ หรือ top N
if MIN_SALES_THRESHOLD:
    popular_products = product_sales[product_sales >= MIN_SALES_THRESHOLD].index
    print(f"✓ Products with sales >= {MIN_SALES_THRESHOLD}: {len(popular_products):,}")
else:
    popular_products = product_sales.head(TOP_N_PRODUCTS).index
    print(f"✓ Top {TOP_N_PRODUCTS} products: {len(popular_products):,}")

weekly_data = weekly_data[weekly_data['product_id'].isin(popular_products)].copy()
print(f"✓ Weekly records (filtered): {len(weekly_data):,}")
print(f"✓ Unique products in dataset: {weekly_data['product_id'].nunique():,}")

# แสดงตัวอย่างสินค้า Top 10
print("\n📦 Top 10 Fashion Products by Total Sales:")
top_products_info = weekly_data.groupby(['product_id', 'product_category_name_english'])['QuantitySold'].sum().sort_values(ascending=False).head(10)
for (pid, cat), qty in top_products_info.items():
    print(f"  {pid[:30]}... ({cat}): {qty} units")

df = weekly_data.copy()
df['AverageSellingPrice'] = df.groupby('product_id')['AverageSellingPrice'].ffill().bfill()
df['AverageSellingPrice'] = df['AverageSellingPrice'].fillna(df['AverageSellingPrice'].median())

for col in ['Weight_g_Mean', 'Length_cm_Mean', 'Height_cm_Mean', 'Width_cm_Mean']:
    df[col] = df.groupby('product_id')[col].ffill().bfill()
    df[col] = df[col].fillna(0)

print(f"✓ Processing time: {time.time() - start_time:.1f}s")

print("\n=== Feature Engineering (Product-Level) ===")
df = df.sort_values(by=['product_id', 'Date'])

le_cat = LabelEncoder()
df['category_encoded'] = le_cat.fit_transform(df['product_category_name_english'])

le_product = LabelEncoder()
df['product_encoded'] = le_product.fit_transform(df['product_id'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
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

is_weekend_series = (df['Date'].dt.dayofweek.isin([5, 6])).astype(int)
df['IsWeekend'] = is_weekend_series

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
# PREPARE TRAIN/TEST (Product-Level Model)
# =============================================================================
print("\n=== Preparing Train/Test Split (Product-Level) ===")
feature_cols = [
    'product_encoded', 'category_encoded', 'seller_state_encoded',
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
X_test_scaled  = scaler_X.transform(X_test)

# =============================================================================
# TRAIN + COMPARE MULTIPLE MODELS
# =============================================================================
def _evaluate_and_record(name, model_obj, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model_obj.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred_tr = model_obj.predict(X_tr)
    y_pred_te = model_obj.predict(X_te)

    res = evaluate_model(y_tr, y_pred_tr, y_te, y_pred_te)
    row = {
        'model': name,
        'train_mae': res['train_mae'],
        'test_mae':  res['test_mae'],
        'train_rmse': res['train_rmse'],
        'test_rmse':  res['test_rmse'],
        'train_r2':  res['train_r2'],
        'test_r2':   res['test_r2'],
        'test_mape': res['test_mape'],
        'overfit_gap_r2': res['train_r2'] - res['test_r2'],
        'train_time_sec': float(train_time)
    }
    return row, y_pred_te, model_obj

print("\n" + "="*90)
print("TRAIN & EVALUATE ALL MODELS (Fashion Products Only - fashion_*)")
print("="*90)

models = {
    'LightGBM': LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=7,
        num_leaves=31, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
}
if _HAS_XGB:
    models['XGBoost'] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )

results, preds, fitted_models = [], {}, {}
for name, mdl in models.items():
    print(f"\n=== {name} ===")
    row, ypred, fitted = _evaluate_and_record(name, mdl, X_train_scaled, y_train, X_test_scaled, y_test)
    results.append(row)
    preds[name] = ypred
    fitted_models[name] = fitted

cmp_df = pd.DataFrame(results).sort_values(by=['test_r2', 'test_rmse'], ascending=[False, True])

print("\n" + "="*90)
print("MODEL COMPARISON TABLE")
print("="*90)
print(
    cmp_df[['model','test_mae','test_rmse','test_r2','test_mape','overfit_gap_r2','train_time_sec']]
      .to_string(index=False, formatters={
          'test_mae': '{:.4f}'.format,
          'test_rmse': '{:.4f}'.format,
          'test_r2': '{:.4f}'.format,
          'test_mape': (lambda v: 'NA' if pd.isna(v) else f'{v:.2f}'),
          'overfit_gap_r2': '{:+.4f}'.format,
          'train_time_sec': '{:.2f}'.format,
      })
)

best_name = cmp_df.iloc[0]['model']
best_model = fitted_models[best_name]
print(f"\n✓ Best model by R² then RMSE: {best_name}")

# วน Loop เพื่อพล็อตกราฟของทุกโมเดล
print("\n" + "="*90)
print("PLOTTING EVALUATION FOR ALL MODELS")
print("="*90)

for model_name, y_pred in preds.items():
    print(f"\nGenerating plots for {model_name}...")
    save_path = f'/mnt/user-data/outputs/eval_fashion_{str(model_name).lower().replace(" ", "_")}.png'
    plot_simple_evaluation(y_test, y_pred, model_name=f"{model_name} (Fashion)", save_path=save_path)

print("\nRunning cross-validation on a sample with the best model...")
sample_size = min(10000, len(X_train_scaled))
sample_idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)
cross_validate_simple(best_model, X_train_scaled[sample_idx], y_train[sample_idx], cv=3)

# =============================================================================
# PRICE OPTIMIZATION (Product-Level) - ใช้ best_model
# =============================================================================
print("\n" + "="*90)
print("PRICE OPTIMIZATION (PRODUCT-LEVEL) - Fashion Items (fashion_* only)")
print("="*90)

df_train = df.iloc[:split_idx].copy()

# เลือก Top N fashion products ที่มียอดขายสูงสุด
target_products = df_train.groupby('product_id')['QuantitySold'].sum().nlargest(NUM_PRODUCTS_TO_OPTIMIZE).index.values
print(f"✓ Optimizing {NUM_PRODUCTS_TO_OPTIMIZE} top fashion products")

# ดึงข้อมูลของแต่ละสินค้าเพื่อใช้ในการ optimize
base_data = df_train.groupby('product_id').last()

# แสดงข้อมูลสินค้าที่จะ optimize
print("\n📦 Products to Optimize:")
for i, prod_id in enumerate(target_products):
    prod_info = df_train[df_train['product_id'] == prod_id].iloc[-1]
    category = prod_info['product_category_name_english']
    avg_price = df_train[df_train['product_id'] == prod_id]['AverageSellingPrice'].mean()
    total_sales = df_train[df_train['product_id'] == prod_id]['QuantitySold'].sum()
    print(f"  {i+1}. {prod_id[:30]}... ({category})")
    print(f"     Avg Price: R${avg_price:.2f}, Total Sales: {total_sales:.0f} units")

price_stats = df_train[df_train['product_id'].isin(target_products)].groupby('product_id')['AverageSellingPrice'].agg(['mean', 'std'])

def optimize_price_grid(model, prod_id, price_bounds, cost, scaler, feature_cols, product_name=""):
    """Optimize price for a specific product using grid search"""
    prices = np.linspace(price_bounds[0], price_bounds[1], PRICE_GRID_POINTS)
    base_features = base_data.loc[prod_id][feature_cols].values

    best_price, best_profit = None, -np.inf
    price_idx = feature_cols.index('AverageSellingPrice')

    for price in prices:
        features_unscaled = base_features.copy()
        features_unscaled[price_idx] = price

        # อัพเดท lag features
        if 'AverageSellingPrice_Lag_1' in feature_cols:
            lag_idx = feature_cols.index('AverageSellingPrice_Lag_1')
            features_unscaled[lag_idx] = base_features[price_idx]

        features_scaled = scaler.transform(features_unscaled.reshape(1, -1))
        predicted_qty = model.predict(features_scaled)[0]
        predicted_qty = max(0, round(float(predicted_qty)))

        profit = (price - cost) * predicted_qty
        if profit > best_profit:
            best_profit, best_price = profit, price

    return best_price, best_profit

optimization_results = {}

for i, prod_id in enumerate(target_products):
    prod_info = df_train[df_train['product_id'] == prod_id].iloc[-1]
    product_name = f"{prod_info['product_category_name_english']}"
    
    print(f"\n{'='*70}")
    print(f"Optimizing Product {i+1}/{len(target_products)}")
    print(f"Product ID: {prod_id[:40]}...")
    print(f"Category: {product_name}")
    
    mean_price = price_stats.loc[prod_id, 'mean']
    price_bounds = (mean_price * 0.7, mean_price * 1.3)
    cost = PRODUCT_COSTS[i]

    print(f"Current avg price: R${mean_price:.2f}")
    print(f"Price range: R${price_bounds[0]:.2f} - R${price_bounds[1]:.2f}")
    print(f"Cost: R${cost:.2f}")

    t0 = time.time()
    optimal_price, max_profit = optimize_price_grid(
        best_model, prod_id, price_bounds, cost, scaler_X, feature_cols, product_name
    )
    opt_time = time.time() - t0

    optimization_results[prod_id] = {
        'product_name': product_name,
        'optimal_price': optimal_price,
        'max_profit': max_profit,
        'current_price': mean_price,
        'time': opt_time
    }

    print(f"✓ Optimal price: R${optimal_price:.2f}")
    print(f"✓ Expected profit: R${max_profit:,.2f}")
    print(f"✓ Price change: {(optimal_price/mean_price - 1)*100:+.1f}%")
    print(f"✓ Optimization time: {opt_time:.3f}s")

# =============================================================================
# SUMMARY (Product-Level)
# =============================================================================
print("\n" + "="*70)
print("OPTIMIZATION SUMMARY - FASHION PRODUCTS (fashion_* only)")
print("="*70)

total_profit = 0
for prod_id, result in optimization_results.items():
    print(f"\n📦 Product: {prod_id[:40]}...")
    print(f"   Category: {result['product_name']}")
    print(f"   Current Price: R${result['current_price']:.2f}")
    print(f"   Optimal Price: R${result['optimal_price']:.2f} ({result['optimal_price']/result['current_price']*100-100:+.1f}%)")
    print(f"   Expected Profit: R${result['max_profit']:,.2f}")
    total_profit += result['max_profit']

print(f"\n{'='*70}")
print(f"TOTAL EXPECTED PROFIT (Fashion Products): R${total_profit:,.2f}")
print(f"{'='*70}")

print("\n✓ Script finished successfully!")
print(f"✓ Model trained on {len(popular_products):,} fashion_* products")
print(f"✓ {NUM_PRODUCTS_TO_OPTIMIZE} products optimized")