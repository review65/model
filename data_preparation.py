# data_preparation.py
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    โหลดข้อมูลจาก Amazon Sale Report.csv, ทำความสะอาด, และเตรียมข้อมูล
    โดยจะทำการรวม (Aggregate) ข้อมูลเป็นรายวันต่อ SKU
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
    df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    df['is_on_promotion'] = df['promotion-ids'].notna().astype(int)
    df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')

    # --- 3. Data Aggregation ---
    daily_sales = df.groupby(['Date', 'SKU', 'Category', 'Size']).agg(
        Total_Qty=('Qty', 'sum'),
        Avg_Price=('Price', 'mean')
        
    ).reset_index()

    # ( === ส่วนที่เพิ่มเข้ามาใหม่ทั้งหมด === )
    # --- 4. สร้าง Lag Features (สำคัญมาก!) ---
    # เราต้องเรียงข้อมูลตาม SKU และ วันที่ ก่อน เพื่อให้ Lag ถูกต้อง
    print("Creating lag features...")
    daily_sales = daily_sales.sort_values(by=['SKU', 'Date'])
    
    # สร้าง feature ยอดขายย้อนหลัง 1 วัน (ของ SKU เดียวกัน)
    daily_sales['Qty_lag_1'] = daily_sales.groupby('SKU')['Total_Qty'].shift(1)
    
    # สร้าง feature ยอดขายย้อนหลัง 7 วัน (ของ SKU เดียวกัน)
    daily_sales['Qty_lag_7'] = daily_sales.groupby('SKU')['Total_Qty'].shift(7)

    # สร้าง feature ค่าเฉลี่ยเคลื่อนที่ 7 วัน (min_periods=1 เพื่อให้มีค่าตั้งแต่แรกๆ)
    daily_sales['Qty_roll_mean_7'] = daily_sales.groupby('SKU')['Total_Qty'].shift(1).rolling(window=7, min_periods=1).mean()

    # เติมค่าว่าง (NaN) ที่เกิดขึ้นในช่วงแรกๆ (เพราะยังไม่มีย้อนหลัง) ด้วย 0
    daily_sales = daily_sales.fillna(0)
    print("Lag features created.")
    # ( === สิ้นสุดส่วนที่เพิ่มเข้ามา === )


    # --- 5. Feature Engineering on Aggregated Data ---
    daily_sales = daily_sales.sort_values(by='Date') 
    
    daily_sales['day_of_week'] = daily_sales['Date'].dt.dayofweek
    daily_sales['month'] = daily_sales['Date'].dt.month
    daily_sales['week_of_year'] = daily_sales['Date'].dt.isocalendar().week.astype(int)
    
    # --- เพิ่ม Feature วันหยุด (ตัวอย่างง่ายๆ) ---
    daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int) # 1 ถ้าเป็นเสาร์-อาทิตย์
    daily_sales['is_month_end'] = daily_sales['Date'].dt.is_month_end.astype(int) # 1 ถ้าเป็นวันสิ้นเดือน (คนมักช้อป)
    
    for col in ['SKU', 'Category', 'Size']:
    # ...
        daily_sales[col] = daily_sales[col].astype('category').cat.codes

    print(f"Data preprocessed successfully. Total records: {len(daily_sales)}")
    print("Aggregated data sample:")
    print(daily_sales.head())
    
    return daily_sales