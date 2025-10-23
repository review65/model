# aggregate_weekly.py
"""
สคริปต์แปลงข้อมูลจาก Daily เป็น Weekly
รันไฟล์นี้แยกต่างหากเพื่อสร้าง Weekly CSV
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("AGGREGATE DAILY DATA TO WEEKLY")
print("=" * 70)

# --- 1. Load Original CSV ---
csv_path = input("Enter path to Amazon Sale Report.csv: ").strip().strip('"')

print(f"\nLoading data from: {csv_path}")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Original records: {len(df)}")

# --- 2. Basic Cleaning ---
print("\nCleaning data...")
cols_to_keep = ['Date', 'Status', 'SKU', 'Category', 'Size', 'Qty', 'Amount']
if 'promotion-ids' in df.columns:
    cols_to_keep.append('promotion-ids')

df = df[cols_to_keep]

valid_statuses = ['Shipped', 'Shipped - Delivered to Buyer']
df = df[df['Status'].isin(valid_statuses)].copy()

df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df.dropna(subset=['Qty', 'Amount'], inplace=True)

df = df[df['Qty'] > 0]
df = df[df['Amount'] > 0]

df['Price'] = df['Amount'] / df['Qty']
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
df.dropna(subset=['Date'], inplace=True)

if 'promotion-ids' in df.columns:
    df['is_on_promotion'] = df['promotion-ids'].notna().astype(int)
else:
    df['is_on_promotion'] = 0

print(f"After cleaning: {len(df)} records")

# --- 3. Aggregate to Weekly ---
print("\nAggregating to weekly...")

# สร้าง Week column (ISO week)
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.isocalendar().week

weekly_data = df.groupby(['Year', 'Week', 'SKU', 'Category', 'Size']).agg(
    Total_Qty=('Qty', 'sum'),
    Total_Amount=('Amount', 'sum'),
    Avg_Price=('Price', 'mean'),
    Max_Price=('Price', 'max'),
    Min_Price=('Price', 'min'),
    Std_Price=('Price', 'std'),
    Has_Promotion=('is_on_promotion', 'max'),
    Transaction_Count=('Qty', 'count'),
    Start_Date=('Date', 'min'),
    End_Date=('Date', 'max')
).reset_index()

weekly_data['Std_Price'].fillna(0, inplace=True)

print(f"Weekly records: {len(weekly_data)}")

# --- 4. Create Week Identifier ---
weekly_data['Week_ID'] = (
    weekly_data['Year'].astype(str) + '-W' + 
    weekly_data['Week'].astype(str).str.zfill(2)
)

# ใช้ Start_Date เป็น Date หลัก
weekly_data['Date'] = weekly_data['Start_Date']

# --- 5. Statistics ---
print("\n" + "=" * 70)
print("WEEKLY DATA STATISTICS")
print("=" * 70)
print(f"Total weeks: {weekly_data['Week_ID'].nunique()}")
print(f"Total SKUs: {weekly_data['SKU'].nunique()}")
print(f"Date range: {weekly_data['Date'].min()} to {weekly_data['Date'].max()}")

print(f"\nTotal_Qty statistics:")
print(f"  Mean:   {weekly_data['Total_Qty'].mean():.2f}")
print(f"  Median: {weekly_data['Total_Qty'].median():.2f}")
print(f"  Std:    {weekly_data['Total_Qty'].std():.2f}")
print(f"  Min:    {weekly_data['Total_Qty'].min():.2f}")
print(f"  Max:    {weekly_data['Total_Qty'].max():.2f}")

zero_weeks = (weekly_data['Total_Qty'] == 0).sum()
print(f"  Zero weeks: {zero_weeks} ({zero_weeks/len(weekly_data)*100:.1f}%)")

# --- 6. Save ---
output_path = csv_path.replace('.csv', '_weekly.csv')
weekly_data.to_csv(output_path, index=False)

print(f"\n✅ Weekly data saved to: {output_path}")
print("\n💡 Next steps:")
print("   1. Update main.py to use the weekly CSV")
print(f"      DATA_FILE = r'{output_path}'")
print("   2. Run main.py again")
print("   3. R² should improve significantly!")

print("\n" + "=" * 70)
print("Sample of weekly data:")
print("=" * 70)
print(weekly_data.head(10))
