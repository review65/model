# compare_models.py
"""
สคริปต์เปรียบเทียบโมเดลเดิมกับโมเดลใหม่
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ตั้งค่า Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def calculate_metrics(y_true, y_pred):
    """คำนวณ Metrics ทั้งหมด"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (ป้องกัน division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

def plot_comparison(old_metrics, new_metrics, save_path='/mnt/user-data/outputs/'):
    """
    สร้างกราฟเปรียบเทียบ
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison: Old vs Improved', 
                 fontsize=16, fontweight='bold')
    
    metrics_names = ['MAE', 'RMSE', 'R²', 'MAPE']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_names)):
        old_val = old_metrics[metric]
        new_val = new_metrics[metric]
        
        # สำหรับ R² ค่าสูงกว่าดีกว่า, ส่วน metrics อื่นค่าต่ำกว่าดีกว่า
        if metric == 'R²':
            improvement = ((new_val - old_val) / (old_val + 1e-6)) * 100
            better = new_val > old_val
        else:
            improvement = ((old_val - new_val) / (old_val + 1e-6)) * 100
            better = new_val < old_val
        
        # Plot bars
        bars = ax.bar(['Old Model', 'Improved Model'], 
                      [old_val, new_val],
                      color=['#ff6b6b' if not better else '#95e1d3', 
                             '#4ecdc4' if better else '#ff6b6b'])
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}' if metric == 'R²' else f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} - {"↑" if better else "↓"} {abs(improvement):.1f}%',
                    color='green' if better else 'red',
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Comparison chart saved to: {save_path}model_comparison.png")
    plt.close()

def plot_predictions_comparison(y_true, y_pred_old, y_pred_new, 
                                 save_path='/mnt/user-data/outputs/'):
    """
    เปรียบเทียบ Predictions
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Old Model
    axes[0].scatter(y_true, y_pred_old, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    r2_old = r2_score(y_true, y_pred_old)
    axes[0].set_xlabel('Actual Demand', fontweight='bold')
    axes[0].set_ylabel('Predicted Demand', fontweight='bold')
    axes[0].set_title(f'Old Model (R²={r2_old:.4f})', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Improved Model
    axes[1].scatter(y_true, y_pred_new, alpha=0.5, s=20, color='green')
    axes[1].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    r2_new = r2_score(y_true, y_pred_new)
    axes[1].set_xlabel('Actual Demand', fontweight='bold')
    axes[1].set_ylabel('Predicted Demand', fontweight='bold')
    axes[1].set_title(f'Improved Model (R²={r2_new:.4f})', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals Comparison
    residuals_old = y_true - y_pred_old
    residuals_new = y_true - y_pred_new
    
    axes[2].hist(residuals_old, bins=50, alpha=0.6, label='Old Model', color='red')
    axes[2].hist(residuals_new, bins=50, alpha=0.6, label='Improved Model', color='green')
    axes[2].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Residual (Actual - Predicted)', fontweight='bold')
    axes[2].set_ylabel('Frequency', fontweight='bold')
    axes[2].set_title('Residuals Distribution', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}predictions_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Predictions comparison saved to: {save_path}predictions_comparison.png")
    plt.close()

def generate_comparison_report(old_metrics, new_metrics, 
                                save_path='/mnt/user-data/outputs/'):
    """
    สร้างรายงานเปรียบเทียบแบบ Text
    """
    report = []
    report.append("=" * 70)
    report.append("MODEL PERFORMANCE COMPARISON REPORT")
    report.append("=" * 70)
    report.append("")
    
    report.append("📊 METRICS COMPARISON:")
    report.append("-" * 70)
    report.append(f"{'Metric':<15} {'Old Model':<20} {'Improved Model':<20} {'Change':<15}")
    report.append("-" * 70)
    
    for metric in ['MAE', 'RMSE', 'R²', 'MAPE']:
        old_val = old_metrics[metric]
        new_val = new_metrics[metric]
        
        if metric == 'R²':
            change = new_val - old_val
            symbol = '↑' if change > 0 else '↓'
            improvement = f"{symbol} {abs(change):.4f}"
        else:
            change = old_val - new_val
            symbol = '↑' if change > 0 else '↓'
            pct = (change / (old_val + 1e-6)) * 100
            improvement = f"{symbol} {abs(change):.2f} ({pct:.1f}%)"
        
        report.append(f"{metric:<15} {old_val:<20.4f} {new_val:<20.4f} {improvement:<15}")
    
    report.append("-" * 70)
    report.append("")
    
    # Overall Assessment
    r2_improvement = ((new_metrics['R²'] - old_metrics['R²']) / 
                      (old_metrics['R²'] + 1e-6)) * 100
    
    report.append("🎯 OVERALL ASSESSMENT:")
    report.append("-" * 70)
    
    if new_metrics['R²'] > old_metrics['R²']:
        report.append(f"✅ Model performance IMPROVED by {r2_improvement:.1f}%")
        report.append(f"   R² increased from {old_metrics['R²']:.4f} to {new_metrics['R²']:.4f}")
    else:
        report.append(f"❌ Model performance DECREASED by {abs(r2_improvement):.1f}%")
        report.append(f"   R² decreased from {old_metrics['R²']:.4f} to {new_metrics['R²']:.4f}")
    
    report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS:")
    report.append("-" * 70)
    
    if new_metrics['R²'] < 0.5:
        report.append("⚠️  R² is still below 0.5. Consider:")
        report.append("   - Checking for data quality issues")
        report.append("   - Adding more relevant features")
        report.append("   - Trying different model architectures")
        report.append("   - Increasing sequence length")
        report.append("   - Removing outliers")
    elif new_metrics['R²'] < 0.7:
        report.append("⚠️  R² is between 0.5-0.7. Room for improvement:")
        report.append("   - Fine-tune hyperparameters")
        report.append("   - Add more interaction features")
        report.append("   - Try ensemble methods")
    else:
        report.append("✅ Excellent performance! R² > 0.7")
        report.append("   - Model is ready for production")
        report.append("   - Continue monitoring performance")
    
    report.append("")
    report.append("=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    with open(f'{save_path}comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✅ Comparison report saved to: {save_path}comparison_report.txt")

# Example usage:
if __name__ == "__main__":
    print("=" * 70)
    print("MODEL COMPARISON TOOL")
    print("=" * 70)
    print("\nThis script helps you compare old and improved model performance.")
    print("\nUsage:")
    print("------")
    print("1. Train both models and save predictions")
    print("2. Import this module:")
    print("   from compare_models import calculate_metrics, plot_comparison")
    print("3. Calculate metrics:")
    print("   old_metrics = calculate_metrics(y_true, y_pred_old)")
    print("   new_metrics = calculate_metrics(y_true, y_pred_new)")
    print("4. Generate comparison:")
    print("   plot_comparison(old_metrics, new_metrics)")
    print("   plot_predictions_comparison(y_true, y_pred_old, y_pred_new)")
    print("   generate_comparison_report(old_metrics, new_metrics)")
    print("\n" + "=" * 70)
    
    # Example with dummy data
    print("\n🔧 Running example with dummy data...")
    
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.uniform(10, 100, n_samples)
    
    # Simulate old model (poor performance)
    y_pred_old = y_true + np.random.normal(0, 30, n_samples)
    y_pred_old = np.maximum(0, y_pred_old)
    
    # Simulate new model (better performance)
    y_pred_new = y_true + np.random.normal(0, 10, n_samples)
    y_pred_new = np.maximum(0, y_pred_new)
    
    old_metrics = calculate_metrics(y_true, y_pred_old)
    new_metrics = calculate_metrics(y_true, y_pred_new)
    
    plot_comparison(old_metrics, new_metrics)
    plot_predictions_comparison(y_true, y_pred_old, y_pred_new)
    generate_comparison_report(old_metrics, new_metrics)
    
    print("\n✅ Example completed! Check the output files.")
