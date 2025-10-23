# data_preparation.py (IMPROVED VERSION)
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    โหลดข้อมูลจาก Amazon Sale Report.csv, ทำความสะอาด, และเตรียมข้อมูล
    พร้อม Feature Engineering ที่ดีขึ้น
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path, low_memory=False)

    # --- 1. Data Cleaning ---
    cols_to_keep = ['Date', 'Status', 'SKU', 'Category', 'Size', 'Qty', 'Amount', 'promotion-ids']
    df = df[cols_to_keep]

    valid_statuses = ['Shipped', 'Shipped - Delivered to Buyer']
    df = df[df['Status'].isin(valid_statuses)].copy()

    df.dropna(subset=['Amount', 'Qty'], inplace=True)
    
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df.dropna(subset=['Qty', 'Amount'], inplace=True)

    df = df[df['Qty'] > 0]
    df = df[df['Amount'] > 0]

    # --- 2. Feature Creation ---
    df['Price'] = df['Amount'] / df['Qty']
    df['is_on_promotion'] = df['promotion-ids'].notna().astype(int)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    # --- 3. Data Aggregation ---
    daily_sales = df.groupby(['Date', 'SKU', 'Category', 'Size']).agg(
        Total_Qty=('Qty', 'sum'),
        Avg_Price=('Price', 'mean'),
        Has_Promotion=('is_on_promotion', 'max'),
        Max_Price=('Price', 'max'),  # NEW
        Min_Price=('Price', 'min'),  # NEW
        Std_Price=('Price', 'std')   # NEW
    ).reset_index()
    
    daily_sales['Std_Price'].fillna(0, inplace=True)  # กรณีที่มี transaction เดียว

    # --- 4. Sort และเติมวันที่ขาดหาย ---
    print("Filling missing dates for each SKU...")
    daily_sales = daily_sales.sort_values(by=['SKU', 'Date'])
    
    # สร้าง Complete Date Range สำหรับแต่ละ SKU
    complete_data = []
    for sku in daily_sales['SKU'].unique():
        sku_data = daily_sales[daily_sales['SKU'] == sku].copy()
        
        # หา Date Range ของ SKU นี้
        date_range = pd.date_range(
            start=sku_data['Date'].min(),
            end=sku_data['Date'].max(),
            freq='D'
        )
        
        # สร้าง DataFrame ที่มีวันครบ
        complete_dates = pd.DataFrame({'Date': date_range})
        complete_dates['SKU'] = sku
        
        # Merge กับข้อมูลจริง
        sku_complete = complete_dates.merge(sku_data, on=['Date', 'SKU'], how='left')
        
        # Fill ค่า Category, Size จากค่าแรก
        sku_complete['Category'].fillna(method='ffill', inplace=True)
        sku_complete['Size'].fillna(method='ffill', inplace=True)
        
        # Fill Qty = 0 สำหรับวันที่ไม่มีขาย
        sku_complete['Total_Qty'].fillna(0, inplace=True)
        sku_complete['Has_Promotion'].fillna(0, inplace=True)
        
        # Fill Price ด้วยค่าล่าสุด
        sku_complete['Avg_Price'].fillna(method='ffill', inplace=True)
        sku_complete['Max_Price'].fillna(method='ffill', inplace=True)
        sku_complete['Min_Price'].fillna(method='ffill', inplace=True)
        sku_complete['Std_Price'].fillna(0, inplace=True)
        
        complete_data.append(sku_complete)
    
    daily_sales = pd.concat(complete_data, ignore_index=True)
    daily_sales = daily_sales.sort_values(by=['SKU', 'Date'])

    # --- 5. สร้าง Lag Features (IMPROVED) ---
    print("Creating lag features...")
    
    # Lag Features - Quantity
    for lag in [1, 3, 7, 14]:
        daily_sales[f'Qty_lag_{lag}'] = daily_sales.groupby('SKU')['Total_Qty'].shift(lag)
    
    # Rolling Features - Quantity
    for window in [3, 7, 14, 30]:
        daily_sales[f'Qty_roll_mean_{window}'] = (
            daily_sales.groupby('SKU')['Total_Qty']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        daily_sales[f'Qty_roll_std_{window}'] = (
            daily_sales.groupby('SKU')['Total_Qty']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
        )
    
    # Lag Features - Price
    for lag in [1, 7]:
        daily_sales[f'Price_lag_{lag}'] = daily_sales.groupby('SKU')['Avg_Price'].shift(lag)
    
    # Lag Features - Promotion
    for lag in [1, 7]:
        daily_sales[f'Promo_lag_{lag}'] = daily_sales.groupby('SKU')['Has_Promotion'].shift(lag)
    
    # Fill NaN
    daily_sales.fillna(0, inplace=True)
    print("Lag features created.")

    # --- 6. Time-based Features (IMPROVED) ---
    daily_sales['day_of_week'] = daily_sales['Date'].dt.dayofweek
    daily_sales['month'] = daily_sales['Date'].dt.month
    daily_sales['week_of_year'] = daily_sales['Date'].dt.isocalendar().week.astype(int)
    daily_sales['day_of_month'] = daily_sales['Date'].dt.day
    daily_sales['quarter'] = daily_sales['Date'].dt.quarter
    
    # Cyclical encoding
    daily_sales['day_of_week_sin'] = np.sin(2 * np.pi * daily_sales['day_of_week'] / 7)
    daily_sales['day_of_week_cos'] = np.cos(2 * np.pi * daily_sales['day_of_week'] / 7)
    daily_sales['month_sin'] = np.sin(2 * np.pi * daily_sales['month'] / 12)
    daily_sales['month_cos'] = np.cos(2 * np.pi * daily_sales['month'] / 12)
    
    # Holiday Features
    daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
    daily_sales['is_month_start'] = (daily_sales['day_of_month'] <= 3).astype(int)
    daily_sales['is_month_end'] = daily_sales['Date'].dt.is_month_end.astype(int)
    
    # --- 7. Interaction Features ---
    daily_sales['price_x_promotion'] = daily_sales['Avg_Price'] * daily_sales['Has_Promotion']
    daily_sales['price_change_pct'] = (
        (daily_sales['Avg_Price'] - daily_sales['Price_lag_1']) / 
        (daily_sales['Price_lag_1'] + 1e-6) * 100
    )
    
    # --- 8. Trend Features ---
    daily_sales['qty_trend_7'] = (
        daily_sales['Qty_lag_1'] - daily_sales['Qty_roll_mean_7']
    )
    daily_sales['qty_trend_30'] = (
        daily_sales['Qty_lag_1'] - daily_sales['Qty_roll_mean_30']
    )
    
    # --- 9. Encode Categorical Features ---
    for col in ['SKU', 'Category', 'Size']:
        daily_sales[col] = daily_sales[col].astype('category').cat.codes

    print(f"Data preprocessed successfully. Total records: {len(daily_sales)}")
    print(f"Number of features: {len(daily_sales.columns)}")
    print("\nFeature columns:")
    print(daily_sales.columns.tolist())
    print("\nAggregated data sample:")
    print(daily_sales.head())
    
    return daily_sales
