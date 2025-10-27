# main_olist.py (Weekly Aggregation + Compare 3 Models + PSO)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import functools
import warnings
from lightgbm import LGBMRegressor

# --- 1. IMPORT MODELS FOR COMPARISON ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# --- 2. IMPORT PRICE OPTIMIZER ---
try:
    from price_optimizer import ParticleSwarmOptimizer
    PSO_ENABLED = True
except ImportError:
    print("Warning: price_optimizer.py not found. Price Optimization section will be skipped.")
    PSO_ENABLED = False

warnings.filterwarnings('ignore', category=FutureWarning) # Suppress pandas warnings

# --- 3. LOAD OLIST DATA ---
print("\n=== Loading Olist Data ===")
try:
    # (โค้ดจะหาไฟล์ในโฟลเดอร์เดียวกัน)
    df_orders = pd.read_csv('olist_orders_dataset.csv')
    df_items = pd.read_csv('olist_order_items_dataset.csv')
    df_products = pd.read_csv('olist_products_dataset.csv')
    df_trans = pd.read_csv('product_category_name_translation.csv')
    # (อาจเพิ่มไฟล์อื่นถ้าต้องการ Feature เพิ่มเติม)
    print("Olist CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not load Olist CSV file - {e}")
    print("Make sure Olist CSV files are in the same folder.")
    exit()

# --- 4. MERGE & PREPARE DATA ---
print("\nMerging and Preparing Olist data...")
# เลือกเฉพาะ Order ที่เสร็จสมบูรณ์
df_orders = df_orders[df_orders['order_status'] == 'delivered'].copy()
# แปลงเป็น datetime
df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])

# Merge orders with items
df = pd.merge(df_orders, df_items, on='order_id', how='inner')

# Merge with products and translation
df = pd.merge(df, df_products, on='product_id', how='left')
df = pd.merge(df, df_trans, on='product_category_name', how='left')

# เลือกคอลัมน์ที่จำเป็น + สร้าง Quantity (order_item_id คือ ลำดับ item ใน order ไม่ใช่ quantity)
df['quantity'] = 1 # Assume each row in order_items is 1 unit sold for simplicity
df_agg = df[[
    'order_purchase_timestamp',
    'product_id',
    'product_category_name_english',
    'price', # ราคาขายต่อหน่วย
    'quantity' # สร้างคอลัมน์ quantity
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm'
]].copy()

# Drop rows with missing category (optional, but simplifies)
df_agg.dropna(subset=['product_category_name_english'], inplace=True)

print(f"Initial transaction records: {len(df_agg)}")

# --- 5. AGGREGATE TO WEEKLY PER PRODUCT ---
print("\nAggregating data to weekly per product...")
df_agg['Date'] = df_agg['order_purchase_timestamp']
df_agg = df_agg.set_index('Date')

# รวมยอดขายและคำนวณราคาเฉลี่ย รายสัปดาห์ ต่อ Product
weekly_data = df_agg.groupby(['product_id', 'product_category_name_english', pd.Grouper(freq='W-MON')]).agg(
    QuantitySold=('quantity', 'count'), # ใช้ count เพราะเราสมมติว่า 1 แถว = 1 ชิ้น
    AverageSellingPrice=('price', 'mean'),
    Weight_g_Mean=('product_weight_g', 'mean'),
    Length_cm_Mean=('product_length_cm', 'mean'),
    Height_cm_Mean=('product_height_cm', 'mean'),
    Width_cm_Mean=('product_width_cm', 'mean')
).reset_index()

# เติม 0 สำหรับสัปดาห์ที่ไม่มีการขาย (สำคัญ!)
# 1. หา Product ทั้งหมด และ Date range ทั้งหมด
all_products = weekly_data[['product_id', 'product_category_name_english']].drop_duplicates()
min_date = weekly_data['Date'].min()
max_date = weekly_data['Date'].max()
# สร้าง Date range รายสัปดาห์ (W-MON)
all_weeks = pd.date_range(start=min_date, end=max_date, freq='W-MON')

# 2. สร้าง DataFrame กรอบเต็ม (Cartesian product)
full_index = pd.MultiIndex.from_product([all_products['product_id'].unique(), all_weeks], names=['product_id', 'Date'])
df_full = pd.DataFrame(index=full_index).reset_index()

# 3. Merge กรอบเต็มกับข้อมูลจริง และเติม 0
weekly_data_filled = pd.merge(df_full, weekly_data, on=['product_id', 'Date'], how='left')
# เติม Category กลับเข้าไป
weekly_data_filled = pd.merge(weekly_data_filled, all_products, on='product_id', how='left', suffixes=('', '_y'))
weekly_data_filled.drop(columns=['product_category_name_english_y'], inplace=True)
# เติม 0 สำหรับ QuantitySold และอาจจะเติมราคาเฉลี่ยด้วย ffill/bfill หรือค่าเฉลี่ยรวม
weekly_data_filled['QuantitySold'] = weekly_data_filled['QuantitySold'].fillna(0)
weekly_data_filled['AverageSellingPrice'] = weekly_data_filled.groupby('product_id')['AverageSellingPrice'].ffill().bfill() # Forward/Backward fill price
weekly_data_filled['AverageSellingPrice'] = weekly_data_filled['AverageSellingPrice'].fillna(weekly_data_filled['AverageSellingPrice'].mean()) # Fill remaining NaNs with global mean

df = weekly_data_filled.copy()
print(f"Weekly aggregated records (filled): {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# --- 6. FEATURE ENGINEERING FOR OLIST (WEEKLY) ---
print("\n=== Creating Olist Weekly Features ===")
df = df.sort_values(by=['product_id', 'Date'])

# Encode Category
print("Encoding categorical features (Category)...")
encoders = {}
le_cat = LabelEncoder()
df['category_encoded'] = le_cat.fit_transform(df['product_category_name_english'])
encoders['category'] = le_cat
encoders['product_id'] = 'Numeric'

# Time Features
print("Creating time features...")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(float)
# --- !! เพิ่มเติม !! ---
df['DayOfWeek'] = df['Date'].dt.dayofweek # 0=Monday, 6=Sunday
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Quarter'] = df['Date'].dt.quarter
# --------------------
df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
df['Week_Sin'] = np.sin(2 * np.pi * df['Week']/52)
df['Week_Cos'] = np.cos(2 * np.pi * df['Week']/52)

# Lag Features (ใช้ QuantitySold และ AverageSellingPrice)
print("Creating lag features...")
df['Qty_Lag_1'] = df.groupby('product_id')['QuantitySold'].shift(1)
df['Price_Lag_1'] = df.groupby('product_id')['AverageSellingPrice'].shift(1)
df['Price_Diff_Lag_1'] = df['AverageSellingPrice'] - df['Price_Lag_1']
# --- !! เพิ่มเติม !! ---
df['Qty_Lag_2'] = df.groupby('product_id')['QuantitySold'].shift(2) # Lag 2 สัปดาห์
df['Qty_Lag_52'] = df.groupby('product_id')['QuantitySold'].shift(52) # Lag 1 ปี
# --------------------

# Rolling Mean Features
print("Creating rolling mean features...")
df['Qty_Roll_Mean_4'] = df.groupby('product_id')['QuantitySold'].shift(1).rolling(window=4, min_periods=1).mean()
df['Price_Roll_Mean_4'] = df.groupby('product_id')['AverageSellingPrice'].shift(1).rolling(window=4, min_periods=1).mean()
# --- !! เพิ่มเติม !! ---
df['Qty_Roll_Std_4'] = df.groupby('product_id')['QuantitySold'].shift(1).rolling(window=4, min_periods=1).std() # Std 4 สัปดาห์
df['Qty_Roll_Mean_12'] = df.groupby('product_id')['QuantitySold'].shift(1).rolling(window=12, min_periods=1).mean() # Mean 12 สัปดาห์
# --------------------

# --- !! เพิ่ม Feature โปรโมชั่นโดยประมาณ !! ---
print("Creating inferred promotion features...")
# คำนวณ % ส่วนลด เทียบกับราคาเฉลี่ย 4 สัปดาห์ก่อนหน้า
df['Discount_Pct_Approx'] = (df['Price_Roll_Mean_4'] - df['AverageSellingPrice']) / (df['Price_Roll_Mean_4'] + 1e-6)
df['Is_Discounted_Approx'] = (df['Discount_Pct_Approx'] > 0.05).astype(int) # สมมติว่าลด > 5% คือโปรโมชั่น
# ----------------------------------------

# Drop rows with NaNs created by Lag/Rolling features (ต้อง Drop เพิ่มเพราะมี Lag_52)
print(f"Shape before dropping NaNs: {df.shape}")
# (ต้องเพิ่ม Lag_2, Lag_52, Roll_Std_4, Roll_Mean_12, Discount_Pct_Approx เข้าไป)
df = df.dropna(subset=['Qty_Lag_1', 'Price_Lag_1', 'Qty_Roll_Mean_4', 'Price_Roll_Mean_4',
                       'Qty_Lag_2', 'Qty_Lag_52', 'Qty_Roll_Std_4', 'Qty_Roll_Mean_12',
                       'Discount_Pct_Approx'])
print(f"Shape after dropping NaNs: {df.shape}")

# --- 7. OLIST FEATURE LIST ---
features = [
    'category_encoded',
    # Time Features
    'Year', 'Month', 'Week', 'Month_Sin', 'Month_Cos', 'Week_Sin', 'Week_Cos',
    'DayOfWeek', 'IsWeekend', 'Quarter', 
    # Price Features
    'AverageSellingPrice', 'Price_Lag_1', 'Price_Diff_Lag_1', 'Price_Roll_Mean_4',
    # Lag/Rolling Demand Features
    'Qty_Lag_1', 'Qty_Roll_Mean_4',
    'Qty_Lag_2', 'Qty_Lag_52', 'Qty_Roll_Std_4', 'Qty_Roll_Mean_12', 
    # Promotion Features (Inferred)
    'Discount_Pct_Approx', 'Is_Discounted_Approx' 
    'Weight_g_Mean', 'Length_cm_Mean', 'Height_cm_Mean', 'Width_cm_Mean'
]
# (คำนวณ NUM_FEATURES ใหม่)
target = 'QuantitySold'
NUM_FEATURES = len(features)
print(f"\nTotal Olist weekly features: {NUM_FEATURES}")
# --- 8. Time-based Train/Validation Split ---
print("\n=== Time-based Train/Validation Split ===")
df = df.sort_values(by=['Date'])

total_weeks = df['Date'].nunique()
train_weeks = int(total_weeks * 0.8) # ใช้ 80% Train
split_date = df['Date'].unique()[train_weeks]

print(f"Splitting data before date: {split_date}")

df_train = df[df['Date'] < split_date].copy()
df_val = df[df['Date'] >= split_date].copy() # ใช้ df_val เป็น Test Set

print(f"Train period: {df_train['Date'].min()} to {df_train['Date'].max()}")
print(f"Validation period: {df_val['Date'].min()} to {df_val['Date'].max()}")

if len(df_train) == 0 or len(df_val) == 0:
    raise ValueError("Data splitting resulted in empty Train or Validation set.")

# --- 9. Fit Scaler ---
print("\n=== Fitting Scaler on Train Set ONLY ===")
X_train_raw = df_train[features].values
y_train_target = df_train[target].values
X_test_raw = df_val[features].values # ใช้ df_val
y_test_raw = df_val[target].values   # ใช้ df_val

# (แยก Scaler สำหรับ y ด้วย ถ้าต้องการ แต่ RF/Linear ไม่จำเป็น)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

print(f"X_train shape (flat): {X_train_scaled.shape}")
print(f"X_test shape (flat): {X_test_scaled.shape}")

# --- 10. Build, Train, and Compare Models ---
print("\n=== Training and Comparing Models (Olist) ===")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5, 
        random_state=42,
        n_jobs=-1,
        verbose=0
    ),
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        random_state=42,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        verbose=True
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42
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
    # ทำให้ค่าพยากรณ์ไม่ติดลบ และเป็นจำนวนเต็ม (สำหรับ Demand)
    predictions_raw = np.maximum(0, predictions_raw)
    predictions_raw = np.round(predictions_raw)


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

# --- 11. Visualization (Plotting the BEST model on Validation Set) ---
print("\nGenerating visualization...")
best_model_preds = results[best_model_name]['preds']
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# สุ่มข้อมูลมาพล็อตบางส่วน ถ้าข้อมูลเยอะเกินไป
plot_indices = np.random.choice(len(y_test_raw), size=min(5000, len(y_test_raw)), replace=False)
plt.scatter(y_test_raw[plot_indices], best_model_preds[plot_indices], alpha=0.1)
plt.plot([y_test_raw.min(), y_test_raw.max()],
         [y_test_raw.min(), y_test_raw.max()], 'r--', lw=2)
plt.xlabel('Actual Quantity Sold')
plt.ylabel('Predicted Quantity Sold')
plt.title(f'Actual vs Predicted ({best_model_name})\nValidation Set (R²={results[best_model_name]["R²"]:.4f})')
plt.grid(True)
plt.subplot(1, 2, 2)
residuals = y_test_raw - best_model_preds
res_min, res_max = residuals.min(), residuals.max()
bins = np.linspace(res_min, res_max, 100)
plt.hist(residuals, bins=bins, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title(f'Residual Distribution ({best_model_name})')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'model_evaluation_olist_COMPARISON.png', dpi=150)
print(f"\nVisualization for best model ({best_model_name}) saved to: model_evaluation_olist_COMPARISON.png")


# --- 12. PRICE OPTIMIZATION SETUP ---
if PSO_ENABLED:
    print("\n" + "="*50)
    print("=== Price Optimization (ACTIVATED for Olist) ===")
    print("="*50)

    # 1. เลือกสินค้าเป้าหมาย (Top 3 จาก Train Set)
    # (เราต้องหา product_id ที่ถูกเข้ารหัสแล้ว ถ้ามีการเข้ารหัส)
    # (ในกรณีนี้ product_id เป็นตัวเลขอยู่แล้ว ใช้ได้เลย)
    product_sales_train = df_train.groupby('product_id')['QuantitySold'].sum()
    top_products_in_train = product_sales_train.nlargest(3).index.values

    target_products = top_products_in_train
    print(f"Optimizing for Top 3 Product IDs from Train Set: {target_products}")

    # !! ต้องกำหนดต้นทุนเอง !! (ใช้ค่าสมมติ)
    # คุณต้องหาต้นทุนจริงของ Product IDs เหล่านี้มาใส่แทน
    PRODUCT_COSTS = np.array([10, 20, 30])
    if len(PRODUCT_COSTS) != len(target_products):
        print("Warning: PRODUCT_COSTS length doesn't match target_products. Using placeholder costs.")
        PRODUCT_COSTS = np.array([10] * len(target_products)) # Placeholder

    NUM_PRODUCTS = len(target_products)

    # 2. ดึง "ข้อมูลฐาน" (สัปดาห์ล่าสุดของ Train Set เพื่อใช้ทำนาย Validation Set)
    # (เราจะใช้ข้อมูลดิบ Unscaled ของสัปดาห์สุดท้ายใน Train Set)
    last_train_date = df_train['Date'].max()
    # หาข้อมูลของสัปดาห์สุดท้ายสำหรับสินค้าเป้าหมาย
    base_data = df_train[(df_train['Date'] == last_train_date) & (df_train['product_id'].isin(target_products))]
    if len(base_data) < NUM_PRODUCTS:
        print("Warning: Could not find base data for all target products in the last training week. Using latest available data.")
        # หาข้อมูลล่าสุดของแต่ละ product แทน
        base_data = df_train[df_train['product_id'].isin(target_products)].sort_values('Date').groupby('product_id').last()

    if len(base_data) == 0:
         print("Error: Could not find any base data for target products. Skipping PSO.")
         target_products = None # Skip PSO loop
    else:
        # ทำให้แน่ใจว่า base_features_unscaled มี NUM_PRODUCTS แถว
        base_features_unscaled_dict = {}
        for prod_id in target_products:
             if prod_id in base_data.index: # กรณี groupby().last()
                 base_features_unscaled_dict[prod_id] = base_data.loc[prod_id][features].values
             elif prod_id in base_data['product_id'].values: # กรณี filter date ตรงๆ
                 base_features_unscaled_dict[prod_id] = base_data[base_data['product_id'] == prod_id][features].iloc[0].values
             else:
                 print(f"Error: Missing base data for product {prod_id}")
                 # อาจจะต้องใช้ค่าเฉลี่ย หรือข้อมูลก่อนหน้า
                 # tạm thời bỏ qua sản phẩm này
                 # target_products = np.delete(target_products, np.where(target_products == prod_id))
                 # NUM_PRODUCTS -= 1
                 # TODO: Handle missing base data more robustly
                 print("Using global average as placeholder - THIS IS NOT ACCURATE")
                 base_features_unscaled_dict[prod_id] = df_train[features].mean().values # ใช้ค่าเฉลี่ยแทน (ไม่แม่นยำ!)


    f_map = {name: idx for idx, name in enumerate(features)}

    # 3. Price Bounds (ต้องปรับให้เหมาะกับราคา Olist)
    # ลองดูช่วงราคาของ Top Products ใน Train set
    price_stats = df_train[df_train['product_id'].isin(target_products)].groupby('product_id')['AverageSellingPrice'].agg(['min', 'max', 'mean'])
    print("\nPrice stats for target products (Train Set):")
    print(price_stats)

    # กำหนด Bounds คร่าวๆ (ควรปรับตามข้อมูลจริง)
    # ตัวอย่าง: ให้ช่วง +/- 30% จากราคาเฉลี่ย
    price_bounds = []
    for prod_id in target_products:
        if prod_id in price_stats.index:
            mean_price = price_stats.loc[prod_id, 'mean']
            lower_bound = max(0.1, mean_price * 0.7) # ต่ำสุด 70%
            upper_bound = mean_price * 1.3        # สูงสุด 130%
            price_bounds.append((lower_bound, upper_bound))
        else:
             print(f"Warning: Missing price stats for product {prod_id}. Using default bounds (5-500).")
             price_bounds.append((5, 500)) # Default กว้างๆ

    print(f"\nUsing Price Bounds: {price_bounds}")


    # 4. สร้าง Objective Function
    def profit_objective_function(prices, model_to_use, current_product_index):
        """
        ฟังก์ชันคำนวณกำไร (สำหรับ Olist)
        """
        prod_id_to_optimize = target_products[current_product_index]
        new_price = prices[0] # PSO จะ optimize ทีละราคา

        # 1. ดึงข้อมูลฐานของ product นี้
        if prod_id_to_optimize not in base_features_unscaled_dict:
             print(f"Error in objective: Missing base data for product {prod_id_to_optimize}")
             return 0 # Return neutral profit (or high cost)

        future_features_unscaled = base_features_unscaled_dict[prod_id_to_optimize].copy()

        # 2. อัปเดตราคาและ features ที่เกี่ยวข้อง
        # Feature 'AverageSellingPrice' คือราคาที่เราจะ optimize
        future_features_unscaled[f_map['AverageSellingPrice']] = new_price

        # อัปเดต Lag/Diff/Roll ที่ขึ้นกับ Price ปัจจุบัน
        # (Lag_1 คือ ราคาของ 'วันนี้' ซึ่งคือ base_features_unscaled)
        today_price = base_features_unscaled_dict[prod_id_to_optimize][f_map['AverageSellingPrice']]
        future_features_unscaled[f_map['Price_Lag_1']] = today_price
        future_features_unscaled[f_map['Price_Diff_Lag_1']] = new_price - today_price

        # Recalculate Price_Roll_Mean_4 including the new price prediction
        # This requires historical prices, which might be complex here.
        # Simplification: Assume Rolling Mean doesn't change drastically in one step OR use the last known rolling mean.
        # Using last known rolling mean from base data:
        future_features_unscaled[f_map['Price_Roll_Mean_4']] = base_features_unscaled_dict[prod_id_to_optimize][f_map['Price_Roll_Mean_4']]
        # TODO: A more accurate approach would involve simulating the rolling mean update.

        # อัปเดต Lag/Roll ของ Qty (ใช้ค่าจาก Base data เพราะเรายังไม่รู้ Qty ของสัปดาห์หน้า)
        # --- !! แก้ไข (FIXED) !! --- ดึง 'QuantitySold' จาก base_data ไม่ใช่ f_map
        last_quantity_sold = 0 # ตั้งค่าเริ่มต้นเป็น 0

        if prod_id_to_optimize in base_data.index:
            # กรณีที่ 1: Fallback (groupby().last()) ทำงาน -> index คือ product_id
            last_quantity_sold = base_data.loc[prod_id_to_optimize]['QuantitySold']
        
        elif 'product_id' in base_data.columns and prod_id_to_optimize in base_data['product_id'].values:
            # กรณีที่ 2: หาสัปดาห์สุดท้ายเจอ (filter by date) -> index คือตัวเลขปกติ
            # เราต้องค้นหาจากคอลัมน์ 'product_id' แทน
            last_quantity_sold = base_data[base_data['product_id'] == prod_id_to_optimize]['QuantitySold'].iloc[0]
        
        else:
            # กรณีที่ 3: หาไม่เจอจริงๆ (ซึ่งไม่ควรเกิดถ้าเป็น Top 3)
            print(f"Warning: Cannot find base QuantitySold for {prod_id_to_optimize}, using 0 for Qty_Lag_1.")
            # last_quantity_sold จะยังคงเป็น 0

        future_features_unscaled[f_map['Qty_Lag_1']] = last_quantity_sold
        future_features_unscaled[f_map['Qty_Roll_Mean_4']] = base_features_unscaled_dict[prod_id_to_optimize][f_map['Qty_Roll_Mean_4']]
        model_input_scaled = scaler_X.transform(future_features_unscaled.reshape(1, -1))

        # 4. Predict Demand
        predicted_qty = model_to_use.predict(model_input_scaled)[0]
        predicted_qty = max(0, round(predicted_qty)) # ทำให้ไม่ติดลบและเป็นจำนวนเต็ม

        # 5. คำนวณกำไร
        cost = PRODUCT_COSTS[current_product_index]
        profit = (new_price - cost) * predicted_qty

        return -profit  # ติดลบเพราะ PSO minimize

    # --- 13. RUN OPTIMIZER FOR ALL 3 MODELS ---
    if target_products is not None and NUM_PRODUCTS > 0:
        optimization_summary = {}

        for model_name, data in results.items():
            print(f"\n--- Optimizing Prices using: {model_name} (R²: {data['R²']:.4f}) ---")
            current_model = data['model_obj']
            all_optimal_prices = []
            total_max_profit = 0

            for i in range(NUM_PRODUCTS):
                prod_id = target_products[i]
                print(f"Optimizing for Product ID: {prod_id}...")

                # สร้าง wrapper ที่ส่ง index ปัจจุบันเข้าไปด้วย
                objective_wrapper = functools.partial(profit_objective_function, model_to_use=current_model, current_product_index=i)

                # PSO สำหรับสินค้าตัวเดียว (ต้องปรับ bounds)
                optimizer = ParticleSwarmOptimizer(
                    objective_function=objective_wrapper,
                    bounds=[price_bounds[i]], # ส่ง bounds แค่ตัวเดียว
                    num_particles= 30, 
                    max_iter=50 ,     
                    verbose=False
                )

                optimal_price_array, max_profit_single = optimizer.optimize()
                optimal_price = optimal_price_array[0] # ผลลัพธ์เป็น array
                max_profit = max_profit_single # แปลงกลับเป็นบวก

                all_optimal_prices.append(optimal_price)
                total_max_profit += max_profit

                print(f"  Optimal Price for Product {prod_id}: {optimal_price:.2f}, Max Profit: {max_profit:,.2f}")

            optimization_summary[model_name] = {'OptimalPrices': all_optimal_prices, 'TotalMaxProfit': total_max_profit}

            print(f"\n{'='*50}")
            print(f"OPTIMIZATION SUMMARY (For {model_name}):")
            print(f"{'='*50}")
            print(f"  Optimal Prices: {np.round(all_optimal_prices, 2)}")
            print(f"  Total Maximum Profit: ฿{total_max_profit:,.2f}")
            print(f"{'='*50}")

else:
    print("\nPrice Optimization skipped due to previous errors.")


print("\nScript finished.")