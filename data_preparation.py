# data_preparation.py
import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    โหลดข้อมูลจาก Amazon Sale Report.csv, ทำความสะอาด, และเตรียมข้อมูล
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path, low_memory=False)

    # --- 1. Data Cleaning ---
    # เลือกเฉพาะคอลัมน์ที่จำเป็น
    cols_to_keep =
    df = df[cols_to_keep]

    # กรองเฉพาะรายการขายที่เสร็จสมบูรณ์
    valid_statuses =
    df = df.isin(valid_statuses)].copy()

    # จัดการกับค่าที่หายไป (Missing Values)
    df.dropna(subset=['Amount', 'Qty'], inplace=True)
    
    # แปลงชนิดข้อมูลให้ถูกต้อง
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df.dropna(subset=['Qty', 'Amount'], inplace=True)

    # กรองรายการที่ไม่มีการขาย (Qty = 0) หรือยอดขายติดลบ
    df = df[df['Qty'] > 0]
    df = df[df['Amount'] > 0]

    # --- 2. Feature Creation ---
    # คำนวณราคาต่อหน่วย (Price per unit)
    df['Price'] = df['Amount'] / df['Qty']

    # แปลง 'Date' เป็น datetime
    df = pd.to_datetime(df, format='%m-%d-%y', errors='coerce')
    df.dropna(subset=, inplace=True)

    # --- 3. Data Aggregation ---
    # รวมข้อมูลเป็นรายวันต่อ SKU เพื่อสร้าง Time Series
    # เราจะหา 'ยอดขายรวม' และ 'ราคาเฉลี่ย' ของแต่ละ SKU ในแต่ละวัน
    daily_sales = df.groupby().agg(
        Total_Qty=('Qty', 'sum'),
        Avg_Price=('Price', 'mean')
    ).reset_index()

    # --- 4. Feature Engineering on Aggregated Data ---
    daily_sales['day_of_week'] = daily_sales.dt.dayofweek
    daily_sales['month'] = daily_sales.dt.month
    daily_sales['week_of_year'] = daily_sales.dt.isocalendar().week.astype(int)
    
    # แปลงคุณลักษณะที่เป็นข้อความ (Categorical)
    # เราจะใช้ Label Encoding เพื่อความง่ายในการจัดการกับ SKU จำนวนมาก
    for col in:
        daily_sales[col] = daily_sales[col].astype('category').cat.codes

    print(f"Data preprocessed successfully. Total records: {len(daily_sales)}")
    print("Aggregated data sample:")
    print(daily_sales.head())
    
    return daily_sales