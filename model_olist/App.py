from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# XGBoost
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
    print("✓ XGBoost imported successfully.")
except ImportError:
    print("⚠ XGBoost not found. Please install it: pip install xgboost")
    _HAS_XGB = False

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from price_optimizer import ParticleSwarmOptimizer
app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
REPORT_FOLDER = 'reports'

# Exchange rate: 1 BRL (Brazilian Real) = 6.8 THB (Thai Baht)
BRL_TO_THB = 6.8

# Fashion categories filter
FASHION_KEYWORDS = ['fashion', 'watches_gifts', 'cool_stuff']

def is_fashion_category(category_name):
    """ตรวจสอบว่าหมวดหมู่เป็น fashion หรือไม่"""
    if pd.isna(category_name):
        return False
    category_lower = str(category_name).lower()
    return any(keyword in category_lower for keyword in FASHION_KEYWORDS)

for folder in [UPLOAD_FOLDER, MODEL_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables to store model and data
model = None
scaler = None
encoders = {}
df_train = None
df_test = None
feature_cols = None
weekly_data = None

# ============================================================================
# ROUTE: Home Page
# ============================================================================

@app.route('/')
def home():
    """Render main dashboard page"""
    return render_template('home.html')


# ============================================================================
# API: Dashboard Metrics
# ============================================================================

@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    """Get main dashboard metrics"""
    try:
        global weekly_data, model, df_test
        
        if weekly_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        # คำนวณ R2 score ถ้าโมเดลมีอยู่
        model_accuracy = 0.0
        if model and df_test is not None:
            try:
                X_test = df_test[feature_cols].values
                y_test = df_test['QuantitySold'].values
                X_test_scaled = scaler.transform(X_test)
                y_pred_test = model.predict(X_test_scaled)
                model_accuracy = r2_score(y_test, y_pred_test)
            except Exception as e:
                print(f"Error calculating R2: {e}")
                model_accuracy = 0.0

        # Calculate metrics
        total_revenue_brl = (weekly_data['QuantitySold'] * weekly_data['AverageSellingPrice']).sum()
        total_revenue_thb = total_revenue_brl * BRL_TO_THB
        total_products = weekly_data['product_id'].nunique()
        avg_price_brl = weekly_data['AverageSellingPrice'].mean()
        avg_price_thb = avg_price_brl * BRL_TO_THB
        total_sales = weekly_data['QuantitySold'].sum()
        
        # Calculate changes (compare last 4 weeks vs previous 4 weeks)
        recent_data = weekly_data.sort_values('Date').tail(8)
        if len(recent_data) == 8:
            recent_4 = recent_data.tail(4)
            prev_4 = recent_data.head(4)
            
            revenue_change = ((recent_4['QuantitySold'] * recent_4['AverageSellingPrice']).sum() / 
                             ((prev_4['QuantitySold'] * prev_4['AverageSellingPrice']).sum() + 1e-6) - 1) * 100
            
            sales_change = (recent_4['QuantitySold'].sum() / (prev_4['QuantitySold'].sum() + 1e-6) - 1) * 100
        else:
            revenue_change = 0
            sales_change = 0

        metrics = {
            'total_revenue_brl': round(total_revenue_brl, 2),
            'total_revenue_thb': round(total_revenue_thb, 2),
            'total_products': int(total_products),
            'avg_price_brl': round(avg_price_brl, 2),
            'avg_price_thb': round(avg_price_thb, 2),
            'total_sales': int(total_sales),
            'revenue_change': round(revenue_change, 1),
            'sales_change': round(sales_change, 1),
            'model_accuracy': round(model_accuracy, 3) if model_accuracy > 0 else 0.0,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/sales-trend', methods=['GET'])
def get_sales_trend():
    """Get sales trend data for chart"""
    try:
        global weekly_data
        
        if weekly_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        # Get last 12 weeks
        trend_data = weekly_data.sort_values('Date').tail(12).groupby('Date').agg({
            'QuantitySold': 'sum',
            'AverageSellingPrice': 'mean'
        }).reset_index()
        
        return jsonify({
            'dates': trend_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'sales': trend_data['QuantitySold'].tolist(),
            'prices_brl': [round(p, 2) for p in trend_data['AverageSellingPrice'].tolist()],
            'prices_thb': [round(p * BRL_TO_THB, 2) for p in trend_data['AverageSellingPrice'].tolist()]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/top-products', methods=['GET'])
def get_top_products():
    """Get top selling fashion products"""
    try:
        global weekly_data
        
        if weekly_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        # Calculate top products
        top_products = weekly_data.groupby(['product_id', 'product_category_name_english']).agg({
            'QuantitySold': 'sum',
            'AverageSellingPrice': 'mean'
        }).reset_index()
        
        top_products['revenue'] = top_products['QuantitySold'] * top_products['AverageSellingPrice']
        top_products = top_products.sort_values('revenue', ascending=False).head(10)
        
        products = []
        for _, row in top_products.iterrows():
            products.append({
                'product_id': row['product_id'][:20] + '...',
                'category': row['product_category_name_english'],
                'units_sold': int(row['QuantitySold']),
                'avg_price_brl': round(row['AverageSellingPrice'], 2),
                'avg_price_thb': round(row['AverageSellingPrice'] * BRL_TO_THB, 2),
                'revenue_brl': round(row['revenue'], 2),
                'revenue_thb': round(row['revenue'] * BRL_TO_THB, 2)
            })
        
        return jsonify({'products': products})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API: Products (Product-Level, not Category-Level)
# ============================================================================

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all available fashion products with basic info"""
    try:
        global weekly_data
        
        if weekly_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        # Get product summary
        product_summary = weekly_data.groupby(['product_id', 'product_category_name_english']).agg({
            'QuantitySold': 'sum',
            'AverageSellingPrice': 'mean'
        }).reset_index()
        
        product_summary['revenue'] = product_summary['QuantitySold'] * product_summary['AverageSellingPrice']
        product_summary = product_summary.sort_values('revenue', ascending=False)
        
        products = []
        for _, row in product_summary.iterrows():
            products.append({
                'product_id': row['product_id'],
                'product_id_short': row['product_id'][:30] + '...',
                'category': row['product_category_name_english'],
                'total_sales': int(row['QuantitySold']),
                'avg_price_brl': round(row['AverageSellingPrice'], 2),
                'avg_price_thb': round(row['AverageSellingPrice'] * BRL_TO_THB, 2),
                'revenue_brl': round(row['revenue'], 2),
                'revenue_thb': round(row['revenue'] * BRL_TO_THB, 2)
            })
        
        return jsonify({'products': products})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/products/<product_id>/analysis', methods=['GET'])
def get_product_analysis(product_id):
    """Get detailed analysis for a specific product"""
    try:
        global weekly_data
        
        if weekly_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        product_data = weekly_data[weekly_data['product_id'] == product_id]
        
        if len(product_data) == 0:
            return jsonify({'error': 'Product not found'}), 404
        
        # Calculate statistics
        total_sales = int(product_data['QuantitySold'].sum())
        avg_price_brl = float(product_data['AverageSellingPrice'].mean())
        avg_price_thb = avg_price_brl * BRL_TO_THB
        total_revenue_brl = float((product_data['QuantitySold'] * product_data['AverageSellingPrice']).sum())
        total_revenue_thb = total_revenue_brl * BRL_TO_THB
        category = product_data['product_category_name_english'].iloc[0]
        
        details = {
            'product_id': product_id,
            'product_id_short': product_id[:30] + '...',
            'category': category,
            'total_sales': total_sales,
            'avg_price_brl': round(avg_price_brl, 2),
            'avg_price_thb': round(avg_price_thb, 2),
            'min_price_brl': round(float(product_data['AverageSellingPrice'].min()), 2),
            'min_price_thb': round(float(product_data['AverageSellingPrice'].min()) * BRL_TO_THB, 2),
            'max_price_brl': round(float(product_data['AverageSellingPrice'].max()), 2),
            'max_price_thb': round(float(product_data['AverageSellingPrice'].max()) * BRL_TO_THB, 2),
            'total_revenue_brl': round(total_revenue_brl, 2),
            'total_revenue_thb': round(total_revenue_thb, 2),
            'weeks_active': int(len(product_data)),
            'avg_weekly_sales': round(product_data['QuantitySold'].mean(), 1)
        }
        
        # Sales history
        history = product_data.groupby('Date').agg({
            'QuantitySold': 'sum',
            'AverageSellingPrice': 'mean'
        }).sort_index().tail(12)
        
        details['sales_history'] = {
            'dates': history.index.strftime('%Y-%m-%d').tolist(),
            'sales': history['QuantitySold'].tolist(),
            'prices_brl': [round(p, 2) for p in history['AverageSellingPrice'].tolist()],
            'prices_thb': [round(p * BRL_TO_THB, 2) for p in history['AverageSellingPrice'].tolist()]
        }
        
        return jsonify(details)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API: Price Optimization (Product-based)
# ============================================================================

@app.route('/api/optimize/product', methods=['POST'])
def optimize_product_price():
    """Optimize price for a specific product using Particle Swarm Optimization (PSO)"""
    try:
        global model, scaler, weekly_data, feature_cols
        
        if model is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.json
        product_id = data.get('product_id')
        cost_brl = float(data.get('cost', 10.0))
        price_range = float(data.get('price_range', 0.3))
        
        # Get product data
        product_data = weekly_data[weekly_data['product_id'] == product_id].sort_values('Date')
        
        if len(product_data) == 0:
            return jsonify({'error': 'Product not found'}), 404
        
        # Get latest data point for base features
        latest_data = product_data.iloc[-1:].copy()
        current_price_brl = float(latest_data['AverageSellingPrice'].iloc[0])
        category = latest_data['product_category_name_english'].iloc[0]
        
        # Create price bounds for PSO
        lower_bound = current_price_brl * (1 - price_range)
        upper_bound = current_price_brl * (1 + price_range)
        
        # ========== PSO Optimization ==========
        def objective_function(params):
            """Objective function for PSO (minimize negative profit)"""
            test_price = params[0]
            
            # Prepare features
            test_features = latest_data[feature_cols].copy()
            test_features.loc[:, 'AverageSellingPrice'] = test_price
            
            # Scale features
            test_scaled = scaler.transform(test_features)
            
            # Predict demand
            predicted_demand = model.predict(test_scaled)[0]
            predicted_demand = max(0, round(predicted_demand))
            
            # Calculate profit (negative for minimization)
            profit_brl = (test_price - cost_brl) * predicted_demand
            return -profit_brl
        
        # Run PSO
        pso = ParticleSwarmOptimizer(
            objective_function=objective_function,
            bounds=[(lower_bound, upper_bound)],
            num_particles=30,
            max_iter=50,
            verbose=False
        )
        
        best_params, best_value = pso.optimize()
        optimal_price_brl = float(best_params[0])
        max_profit_brl = float(-best_value)  # Convert back to positive
        
        # Calculate expected demand at optimal price
        test_features_opt = latest_data[feature_cols].copy()
        test_features_opt.loc[:, 'AverageSellingPrice'] = optimal_price_brl
        test_scaled_opt = scaler.transform(test_features_opt)
        expected_demand = int(max(0, round(model.predict(test_scaled_opt)[0])))
        
        # ========== Generate visualization curve ==========
        test_prices = np.linspace(lower_bound, upper_bound, 20)
        profits_brl = []
        profits_thb = []
        demands = []
        
        for test_price in test_prices:
            test_features = latest_data[feature_cols].copy()
            test_features.loc[:, 'AverageSellingPrice'] = test_price
            test_scaled = scaler.transform(test_features)
            predicted_demand = model.predict(test_scaled)[0]
            predicted_demand = max(0, round(predicted_demand))
            
            profit_brl = (test_price - cost_brl) * predicted_demand
            profit_thb = profit_brl * BRL_TO_THB
            
            profits_brl.append(float(profit_brl))
            profits_thb.append(float(profit_thb))
            demands.append(int(predicted_demand))
        
        # Calculate current profit for comparison
        current_demand = int(latest_data['QuantitySold'].iloc[0])
        current_profit_brl = (current_price_brl - cost_brl) * current_demand
        current_profit_thb = current_profit_brl * BRL_TO_THB
        max_profit_thb = max_profit_brl * BRL_TO_THB
        
        result = {
            'product_id': product_id,
            'product_id_short': product_id[:30] + '...',
            'category': category,
            'current_price_brl': round(current_price_brl, 2),
            'current_price_thb': round(current_price_brl * BRL_TO_THB, 2),
            'optimal_price_brl': round(optimal_price_brl, 2),
            'optimal_price_thb': round(optimal_price_brl * BRL_TO_THB, 2),
            'price_change_pct': round((optimal_price_brl / (current_price_brl + 1e-6) - 1) * 100, 1),
            'current_demand': current_demand,
            'expected_demand': expected_demand,
            'demand_change_pct': round((expected_demand / (current_demand + 1e-6) - 1) * 100, 1) if current_demand > 0 else 0,
            'current_profit_brl': round(current_profit_brl, 2),
            'current_profit_thb': round(current_profit_thb, 2),
            'expected_profit_brl': round(max_profit_brl, 2),
            'expected_profit_thb': round(max_profit_thb, 2),
            'profit_increase_brl': round(max_profit_brl - current_profit_brl, 2),
            'profit_increase_thb': round(max_profit_thb - current_profit_thb, 2),
            'profit_increase_pct': round((max_profit_brl / (current_profit_brl + 1e-6) - 1) * 100, 1) if current_profit_brl > 0 else 0,
            'optimization_curve': {
                'prices_brl': [round(p, 2) for p in test_prices],
                'prices_thb': [round(p * BRL_TO_THB, 2) for p in test_prices],
                'profits_brl': [round(p, 2) for p in profits_brl],
                'profits_thb': [round(p, 2) for p in profits_thb],
                'demands': demands
            },
            'recommendation': get_price_recommendation(optimal_price_brl, current_price_brl),
            'optimization_method': 'Particle Swarm Optimization (PSO)'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_price_recommendation(optimal_price, current_price):
    """Generate price recommendation text"""
    change_pct = (optimal_price / (current_price + 1e-6) - 1) * 100
    
    if change_pct > 15:
        return f"แนะนำขึ้นราคา {change_pct:.1f}% (ทำทีละน้อย 5-10% ก่อน)"
    elif change_pct > 5:
        return f"แนะนำขึ้นราคาเล็กน้อย {change_pct:.1f}%"
    elif change_pct < -15:
        return f"แนะนำลดราคา {abs(change_pct):.1f}% (ทำโปรโมชั่น)"
    elif change_pct < -5:
        return f"แนะนำลดราคาเล็กน้อย {abs(change_pct):.1f}%"
    else:
        return "ราคาปัจจุบันเหมาะสมแล้ว คงราคาไว้"


# ============================================================================
# API: Model Training (Fashion Products Only)
# ============================================================================

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Train ML model with uploaded data - Fashion products only"""
    try:
        global model, scaler, encoders, df_train, df_test, feature_cols, weekly_data
        
        # Load data from uploaded files
        df_orders = pd.read_csv(f'{UPLOAD_FOLDER}/olist_orders_dataset.csv')
        df_items = pd.read_csv(f'{UPLOAD_FOLDER}/olist_order_items_dataset.csv')
        df_products = pd.read_csv(f'{UPLOAD_FOLDER}/olist_products_dataset.csv')
        df_trans = pd.read_csv(f'{UPLOAD_FOLDER}/product_category_name_translation.csv')
        df_sellers = pd.read_csv(f'{UPLOAD_FOLDER}/olist_sellers_dataset.csv')
        
        start_time = datetime.now()
        
        # Process data
        df_orders = df_orders[df_orders['order_status'] == 'delivered'].copy()
        df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])
        
        df = pd.merge(df_orders, df_items, on='order_id', how='inner')
        df = pd.merge(df, df_products, on='product_id', how='left')
        df = pd.merge(df, df_trans, on='product_category_name', how='left')
        df = pd.merge(df, df_sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
        
        # ===== กรองเฉพาะ Fashion Products =====
        print(f"Records before fashion filter: {len(df):,}")
        df = df[df['product_category_name_english'].apply(is_fashion_category)].copy()
        print(f"Records after fashion filter: {len(df):,}")
        
        if len(df) == 0:
            return jsonify({'status': 'error', 'error': 'No fashion products found in dataset'}), 400
        
        df['seller_state'] = df['seller_state'].fillna('Unknown')
        le_state = LabelEncoder()
        df['seller_state_encoded'] = le_state.fit_transform(df['seller_state'])
        
        df['quantity'] = 1
        df_agg = df[['order_purchase_timestamp', 'product_id', 'product_category_name_english',
                    'seller_state_encoded', 'price', 'quantity', 'product_weight_g',
                    'product_length_cm', 'product_height_cm', 'product_width_cm']].copy()
        
        df_agg.dropna(subset=['product_category_name_english'], inplace=True)
        df_agg['Date'] = df_agg['order_purchase_timestamp']
        df_agg = df_agg.set_index('Date')
        
        # Weekly aggregation (Product-level)
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
        
        # Filter popular products
        MIN_SALES = 10
        product_sales = weekly_data.groupby('product_id')['QuantitySold'].sum()
        popular_products = product_sales[product_sales >= MIN_SALES].index
        weekly_data = weekly_data[weekly_data['product_id'].isin(popular_products)]
        
        # Feature engineering
        weekly_data = weekly_data.sort_values(['product_id', 'Date'])
        
        le_cat = LabelEncoder()
        weekly_data['category_encoded'] = le_cat.fit_transform(weekly_data['product_category_name_english'])
        
        weekly_data['Year'] = weekly_data['Date'].dt.year
        weekly_data['Month'] = weekly_data['Date'].dt.month
        weekly_data['Week'] = weekly_data['Date'].dt.isocalendar().week
        weekly_data['Quarter'] = weekly_data['Date'].dt.quarter
        weekly_data['IsWeekend'] = weekly_data['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Lag features
        weekly_data['QuantitySold_Lag_1'] = weekly_data.groupby('product_id')['QuantitySold'].shift(1)
        weekly_data['QuantitySold_Lag_4'] = weekly_data.groupby('product_id')['QuantitySold'].shift(4)
        weekly_data['AverageSellingPrice_Lag_1'] = weekly_data.groupby('product_id')['AverageSellingPrice'].shift(1)
        weekly_data['AverageSellingPrice_Lag_4'] = weekly_data.groupby('product_id')['AverageSellingPrice'].shift(4)
        
        # Rolling features
        weekly_data['QuantitySold_Roll_Mean_4'] = weekly_data.groupby('product_id')['QuantitySold'].shift(1).rolling(4, min_periods=1).mean().reset_index(0, drop=True)
        weekly_data['AverageSellingPrice_Roll_Mean_4'] = weekly_data.groupby('product_id')['AverageSellingPrice'].shift(1).rolling(4, min_periods=1).mean().reset_index(0, drop=True)
        
        # Difference features
        weekly_data['Price_Diff_Lag_1'] = weekly_data['AverageSellingPrice'] - weekly_data['AverageSellingPrice_Lag_1']
        weekly_data['Qty_Diff_Lag_1'] = weekly_data['QuantitySold'] - weekly_data['QuantitySold_Lag_1']
        
        # Drop NaN
        weekly_data = weekly_data.dropna()
        
        # Feature list
        feature_cols = [
            'category_encoded', 'seller_state_encoded',
            'AverageSellingPrice', 'Weight_g_Mean', 'Length_cm_Mean', 'Height_cm_Mean', 'Width_cm_Mean',
            'Year', 'Month', 'Week', 'Quarter', 'IsWeekend',
            'AverageSellingPrice_Lag_1', 'AverageSellingPrice_Lag_4',
            'QuantitySold_Lag_1', 'QuantitySold_Lag_4',
            'AverageSellingPrice_Roll_Mean_4', 'QuantitySold_Roll_Mean_4',
            'Price_Diff_Lag_1', 'Qty_Diff_Lag_1'
        ]
        
        target = 'QuantitySold'
        
        # Train/test split
        split_idx = int(len(weekly_data) * 0.8)
        df_train = weekly_data.iloc[:split_idx].copy()
        df_test = weekly_data.iloc[split_idx:].copy()
        
        X_train = df_train[feature_cols].values
        y_train = df_train[target].values
        X_test = df_test[feature_cols].values
        y_test = df_test[target].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if not _HAS_XGB:
            return jsonify({'status': 'error', 'error': 'XGBoost library not found. Please install it with: pip install xgboost'}), 500

        print("Training with XGBoost on Fashion products...")
        
        model = XGBRegressor(
            n_estimators=300, 
            learning_rate=0.05, 
            max_depth=7,
            subsample=0.8, 
            colsample_bytree=0.8, 
            reg_lambda=1.0,
            random_state=42, 
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        y_test_safe = y_test.copy()
        y_test_safe[y_test_safe == 0] = 1e-6
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test_safe)) * 100
        
        # Save model
        with open(f'{MODEL_FOLDER}/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(f'{MODEL_FOLDER}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'{MODEL_FOLDER}/encoders.pkl', 'wb') as f:
            pickle.dump({'category': le_cat, 'state': le_state}, f)
        with open(f'{MODEL_FOLDER}/features.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'status': 'success',
            'training_time': round(training_time, 1),
            'data_size': len(weekly_data),
            'num_products': len(popular_products),
            'num_categories': weekly_data['product_category_name_english'].nunique(),
            'product_type': 'Fashion Products Only',
            'metrics': {
                'train_r2': round(train_r2, 4),
                'test_r2': round(test_r2, 4),
                'test_mae': round(test_mae, 2),
                'test_rmse': round(test_rmse, 2),
                'test_mape': round(test_mape, 2)
            },
            'feature_importance': get_feature_importance()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


def get_feature_importance():
    """Get feature importance from trained model"""
    global model, feature_cols
    
    if model is None:
        return []
    
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False).head(10)
    
    return [{
        'feature': row['feature'],
        'importance': round(row['importance'], 2)
    } for _, row in feature_imp.iterrows()]


@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """Get model training status"""
    global model, df_train, df_test
    
    if model is None:
        return jsonify({
            'trained': False,
            'message': 'Model not trained yet'
        })
    
    return jsonify({
        'trained': True,
        'train_samples': len(df_train) if df_train is not None else 0,
        'test_samples': len(df_test) if df_test is not None else 0,
        'num_features': len(feature_cols) if feature_cols else 0,
        'product_type': 'Fashion Products'
    })


# ============================================================================
# API: File Upload
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        uploaded = []
        
        required_files = [
            'olist_orders_dataset.csv',
            'olist_order_items_dataset.csv',
            'olist_products_dataset.csv',
            'product_category_name_translation.csv',
            'olist_sellers_dataset.csv'
        ]
        
        for file in files:
            if file.filename in required_files:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                uploaded.append(file.filename)
        
        # Check if all required files are uploaded
        all_uploaded = all(os.path.exists(f'{UPLOAD_FOLDER}/{f}') for f in required_files)
        
        return jsonify({
            'uploaded': uploaded,
            'all_files_ready': all_uploaded,
            'missing': [f for f in required_files if not os.path.exists(f'{UPLOAD_FOLDER}/{f}')]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Dynamic Pricing System - Fashion Products Only")
    print("=" * 60)
    print(f"Server running on http://localhost:5000")
    print(f"Currency: Brazilian Real (R$) + Thai Baht (฿)")
    print(f"Exchange Rate: 1 BRL = {BRL_TO_THB} THB")
    print(f"Product Focus: Fashion Items Only")
    print(f"Model Level: Product-Level (not Category)")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Model folder: {MODEL_FOLDER}")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)