# 🔍 Debugging Checklist: แก้ปัญหา R² ต่ำ

## ✅ Pre-Training Checklist

### 1. Data Quality
- [ ] ตรวจสอบ Missing Values
  ```python
  print(df.isnull().sum())
  print(f"Missing %: {df.isnull().sum() / len(df) * 100}")
  ```

- [ ] ตรวจสอบ Duplicates
  ```python
  duplicates = df.duplicated().sum()
  print(f"Duplicates: {duplicates}")
  ```

- [ ] ตรวจสอบ Outliers
  ```python
  Q1 = df['Total_Qty'].quantile(0.25)
  Q3 = df['Total_Qty'].quantile(0.75)
  IQR = Q3 - Q1
  outliers = df[(df['Total_Qty'] < Q1 - 1.5*IQR) | 
                (df['Total_Qty'] > Q3 + 1.5*IQR)]
  print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
  ```

- [ ] ตรวจสอบ Data Types
  ```python
  print(df.dtypes)
  print(df.info())
  ```

- [ ] ตรวจสอบ Date Range
  ```python
  print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
  print(f"Total days: {(df['Date'].max() - df['Date'].min()).days}")
  ```

### 2. Target Variable Analysis
- [ ] Plot Distribution
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  plt.figure(figsize=(12, 4))
  
  plt.subplot(1, 3, 1)
  plt.hist(df['Total_Qty'], bins=50, edgecolor='black')
  plt.title('Original Distribution')
  plt.xlabel('Total Qty')
  
  plt.subplot(1, 3, 2)
  plt.hist(np.log1p(df['Total_Qty']), bins=50, edgecolor='black')
  plt.title('Log-transformed Distribution')
  plt.xlabel('Log(Total Qty + 1)')
  
  plt.subplot(1, 3, 3)
  plt.boxplot(df['Total_Qty'])
  plt.title('Boxplot')
  plt.ylabel('Total Qty')
  
  plt.tight_layout()
  plt.show()
  ```

- [ ] Check Statistics
  ```python
  print(df['Total_Qty'].describe())
  print(f"Skewness: {df['Total_Qty'].skew():.2f}")
  print(f"Kurtosis: {df['Total_Qty'].kurtosis():.2f}")
  ```

- [ ] Check Zero Values
  ```python
  zeros = (df['Total_Qty'] == 0).sum()
  print(f"Zero values: {zeros} ({zeros/len(df)*100:.2f}%)")
  ```

### 3. Feature Engineering Validation
- [ ] Check Feature Correlations
  ```python
  # Top correlations with target
  correlations = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
  print("Top 10 features by correlation:")
  print(correlations.head(10))
  
  # Remove low correlation features
  low_corr = correlations[correlations < 0.1]
  print(f"\nLow correlation features (< 0.1): {len(low_corr)}")
  print(low_corr)
  ```

- [ ] Check Multicollinearity
  ```python
  # Correlation matrix
  corr_matrix = df[features].corr().abs()
  
  # Find highly correlated features
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  high_corr = [(column, row, upper.loc[row, column]) 
               for column in upper.columns 
               for row in upper.index 
               if upper.loc[row, column] > 0.9]
  
  print(f"Highly correlated pairs (>0.9): {len(high_corr)}")
  for col, row, corr in high_corr:
      print(f"  {col} <-> {row}: {corr:.3f}")
  ```

- [ ] Check Feature Variance
  ```python
  # Zero or near-zero variance features
  from sklearn.feature_selection import VarianceThreshold
  
  selector = VarianceThreshold(threshold=0.01)
  selector.fit(df[features])
  
  low_var_features = [features[i] for i in range(len(features)) 
                      if not selector.get_support()[i]]
  print(f"Low variance features: {len(low_var_features)}")
  print(low_var_features)
  ```

### 4. Data Split Validation
- [ ] Check Train/Val/Test Split
  ```python
  print(f"Train size: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
  print(f"Val size:   {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
  print(f"Test size:  {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
  ```

- [ ] Check Date Ranges Don't Overlap
  ```python
  assert df_train['Date'].max() < df_val['Date'].min(), "Train-Val overlap!"
  assert df_val['Date'].max() < df_test['Date'].min(), "Val-Test overlap!"
  print("✅ No date overlaps")
  ```

- [ ] Check Target Distribution Across Splits
  ```python
  print("\nTarget statistics by split:")
  print(f"Train - Mean: {df_train[target].mean():.2f}, Std: {df_train[target].std():.2f}")
  print(f"Val   - Mean: {df_val[target].mean():.2f}, Std: {df_val[target].std():.2f}")
  print(f"Test  - Mean: {df_test[target].mean():.2f}, Std: {df_test[target].std():.2f}")
  ```

### 5. Scaling Validation
- [ ] Verify Scaler is Fitted on Train Only
  ```python
  # Check if scaler statistics match train data
  print("Scaler mean (should match train):")
  print(f"  Scaler: {scaler_X.mean_[0]:.4f}")
  print(f"  Train:  {X_train_raw[:, 0].mean():.4f}")
  
  assert np.allclose(scaler_X.mean_, X_train_raw.mean(axis=0)), "Scaler not fitted on train!"
  print("✅ Scaler correctly fitted on train data")
  ```

- [ ] Check Scaled Data Range
  ```python
  print("\nScaled data range:")
  print(f"Train: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
  print(f"Val:   [{X_val_scaled.min():.2f}, {X_val_scaled.max():.2f}]")
  print(f"Test:  [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")
  ```

### 6. Sequence Creation Validation
- [ ] Check Sequence Shapes
  ```python
  print(f"\nSequence shapes:")
  print(f"X_train_seq: {X_train_seq.shape}")
  print(f"y_train_seq: {y_train_seq.shape}")
  print(f"Expected: ({X_train_seq.shape[0]}, {SEQUENCE_LENGTH}, {NUM_FEATURES})")
  
  assert X_train_seq.shape[1] == SEQUENCE_LENGTH, "Wrong sequence length!"
  assert X_train_seq.shape[2] == NUM_FEATURES, "Wrong number of features!"
  print("✅ Sequence shapes correct")
  ```

- [ ] Verify No Data Leakage in Sequences
  ```python
  # Check that no future data is in past sequences
  for i in range(min(5, len(X_train_seq))):
      seq = X_train_seq[i]
      target = y_train_seq[i]
      # Verify sequence is chronological
      # (This is hard to verify programmatically, but visual inspection helps)
  ```

---

## ✅ During Training Checklist

### 7. Monitor Training Progress
- [ ] Check Loss Decrease
  ```python
  # After training
  plt.figure(figsize=(10, 4))
  
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='Train Loss')
  plt.plot(history.history['val_loss'], label='Val Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training History')
  plt.legend()
  plt.grid(True)
  
  plt.subplot(1, 2, 2)
  plt.plot(history.history['mae'], label='Train MAE')
  plt.plot(history.history['val_mae'], label='Val MAE')
  plt.xlabel('Epoch')
  plt.ylabel('MAE')
  plt.title('MAE History')
  plt.legend()
  plt.grid(True)
  
  plt.tight_layout()
  plt.show()
  ```

- [ ] Check for Overfitting
  ```python
  final_train_loss = history.history['loss'][-1]
  final_val_loss = history.history['val_loss'][-1]
  
  if final_val_loss > final_train_loss * 1.5:
      print("⚠️  Possible overfitting detected!")
      print(f"   Train Loss: {final_train_loss:.4f}")
      print(f"   Val Loss:   {final_val_loss:.4f}")
  else:
      print("✅ No obvious overfitting")
  ```

- [ ] Check for Underfitting
  ```python
  if final_train_loss > 0.5:  # Threshold depends on your data
      print("⚠️  Possible underfitting detected!")
      print(f"   Train Loss: {final_train_loss:.4f}")
      print("   Consider:")
      print("   - Increasing model complexity")
      print("   - Training longer")
      print("   - Reducing regularization")
  else:
      print("✅ No obvious underfitting")
  ```

### 8. Check Callbacks
- [ ] Verify Early Stopping Triggered
  ```python
  if len(history.history['loss']) < epochs:
      print(f"✅ Early stopping at epoch {len(history.history['loss'])}")
  else:
      print(f"⚠️  Reached max epochs ({epochs})")
  ```

- [ ] Check Learning Rate Reduction
  ```python
  # If using ReduceLROnPlateau
  # Check logs for learning rate changes
  ```

---

## ✅ Post-Training Checklist

### 9. Prediction Analysis
- [ ] Check Prediction Range
  ```python
  print("\nPrediction range:")
  print(f"Min: {predictions_raw.min():.2f}")
  print(f"Max: {predictions_raw.max():.2f}")
  print(f"Mean: {predictions_raw.mean():.2f}")
  print(f"Std: {predictions_raw.std():.2f}")
  
  print("\nActual range:")
  print(f"Min: {y_test_seq_raw.min():.2f}")
  print(f"Max: {y_test_seq_raw.max():.2f}")
  print(f"Mean: {y_test_seq_raw.mean():.2f}")
  print(f"Std: {y_test_seq_raw.std():.2f}")
  ```

- [ ] Check for Prediction Bias
  ```python
  bias = (predictions_raw - y_test_seq_raw).mean()
  print(f"\nPrediction bias: {bias:.2f}")
  
  if abs(bias) > y_test_seq_raw.mean() * 0.1:
      print("⚠️  Significant prediction bias detected!")
  else:
      print("✅ Low prediction bias")
  ```

- [ ] Analyze Residuals
  ```python
  residuals = y_test_seq_raw - predictions_raw
  
  plt.figure(figsize=(12, 4))
  
  plt.subplot(1, 3, 1)
  plt.scatter(predictions_raw, residuals, alpha=0.5)
  plt.axhline(0, color='red', linestyle='--')
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.title('Residual Plot')
  
  plt.subplot(1, 3, 2)
  plt.hist(residuals, bins=50, edgecolor='black')
  plt.axvline(0, color='red', linestyle='--')
  plt.xlabel('Residuals')
  plt.ylabel('Frequency')
  plt.title('Residual Distribution')
  
  plt.subplot(1, 3, 3)
  from scipy import stats
  stats.probplot(residuals, dist="norm", plot=plt)
  plt.title('Q-Q Plot')
  
  plt.tight_layout()
  plt.show()
  
  # Shapiro-Wilk test for normality
  if len(residuals) < 5000:  # Test is sensitive to large samples
      stat, p = stats.shapiro(residuals)
      print(f"\nShapiro-Wilk test: p-value = {p:.4f}")
      if p > 0.05:
          print("✅ Residuals are approximately normal")
      else:
          print("⚠️  Residuals are not normally distributed")
  ```

### 10. Metrics Analysis
- [ ] Calculate All Metrics
  ```python
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
  from sklearn.metrics import mean_absolute_percentage_error
  
  mae = mean_absolute_error(y_test_seq_raw, predictions_raw)
  mse = mean_squared_error(y_test_seq_raw, predictions_raw)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_test_seq_raw, predictions_raw)
  
  # MAPE (careful with zeros)
  mape = np.mean(np.abs((y_test_seq_raw - predictions_raw) / 
                        (y_test_seq_raw + 1e-6))) * 100
  
  print(f"\nMetrics:")
  print(f"  MAE:  {mae:.2f}")
  print(f"  RMSE: {rmse:.2f}")
  print(f"  R²:   {r2:.4f}")
  print(f"  MAPE: {mape:.2f}%")
  ```

- [ ] Interpret R² Score
  ```python
  if r2 < 0:
      print("❌ R² < 0: Model is worse than predicting mean!")
      print("   Action: Complete model redesign needed")
  elif r2 < 0.3:
      print("❌ R² < 0.3: Very poor performance")
      print("   Action: Check data quality and features")
  elif r2 < 0.5:
      print("⚠️  R² < 0.5: Poor performance")
      print("   Action: Improve features and model")
  elif r2 < 0.7:
      print("⚠️  R² < 0.7: Moderate performance")
      print("   Action: Fine-tune hyperparameters")
  elif r2 < 0.9:
      print("✅ R² < 0.9: Good performance")
  else:
      print("✅ R² ≥ 0.9: Excellent performance!")
      print("   ⚠️  But check for data leakage!")
  ```

### 11. Cross-Validation (Optional but Recommended)
- [ ] Time Series CV
  ```python
  from sklearn.model_selection import TimeSeriesSplit
  
  tscv = TimeSeriesSplit(n_splits=5)
  cv_scores = []
  
  for train_idx, val_idx in tscv.split(X_seq_combined):
      X_train_cv = X_seq_combined[train_idx]
      y_train_cv = y_seq_log_combined[train_idx]
      X_val_cv = X_seq_combined[val_idx]
      y_val_cv = y_seq_raw_combined[val_idx]
      
      # Train and evaluate
      model_cv = build_lstm_model(input_shape)
      model_cv.fit(X_train_cv, y_train_cv, epochs=50, 
                   batch_size=32, verbose=0)
      
      pred_cv = np.expm1(model_cv.predict(X_val_cv).flatten())
      pred_cv = np.maximum(0, pred_cv)
      
      r2_cv = r2_score(y_val_cv, pred_cv)
      cv_scores.append(r2_cv)
      print(f"Fold R²: {r2_cv:.4f}")
  
  print(f"\nCross-validation R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
  ```

---

## 🎯 Action Plan Based on R²

### If R² < 0.3:
1. ✅ Check for data leakage (most common issue)
2. ✅ Verify time-based split
3. ✅ Check target variable distribution
4. ✅ Add more relevant features
5. ✅ Check for missing dates

### If 0.3 ≤ R² < 0.5:
1. ✅ Improve feature engineering
2. ✅ Try different sequence lengths
3. ✅ Tune model architecture
4. ✅ Check for outliers

### If 0.5 ≤ R² < 0.7:
1. ✅ Fine-tune hyperparameters
2. ✅ Try ensemble methods
3. ✅ Add interaction features
4. ✅ Optimize learning rate

### If R² ≥ 0.7:
1. ✅ Verify results (check for data leakage!)
2. ✅ Monitor on new data
3. ✅ Consider deployment

---

## 📝 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Data Leakage | R² too high (>0.95) | Fit scaler on train only |
| Time Leakage | R² decreases on new data | Use time-based split |
| Underfitting | High train & val loss | Increase model complexity |
| Overfitting | Low train, high val loss | Add regularization/dropout |
| Poor Features | Low correlation with target | Engineer better features |
| Missing Dates | Sequences are broken | Fill missing dates with 0 |
| Outliers | High RMSE, low R² | Remove or cap outliers |
| Imbalanced Data | Bias towards high values | Use stratified sampling |

---

**Remember:** Always start with the basics (data quality, no leakage) before 
tuning complex hyperparameters!
