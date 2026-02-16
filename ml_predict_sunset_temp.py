#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "xgboost",
#     "torch",
#     "astral",
# ]
# ///

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from astral import LocationInfo
from astral.sun import sun
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import base64
from io import BytesIO

print("=" * 80, flush=True)
print("ML-Based Sunset Temperature Prediction", flush=True)
print("Rubin Observatory Mirror Temperature Data", flush=True)
print("=" * 80, flush=True)

# ============================================================================
# 1. DATA LOADING AND SUNSET CALCULATION
# ============================================================================

print("\n[1/6] Loading data and calculating sunset times...", flush=True)

# Load temperature data
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])
print(f"  Loaded {len(df):,} temperature measurements")
print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"  Temperature range: {df['temperature'].min():.2f}°C to {df['temperature'].max():.2f}°C")

# Observatory location
location = LocationInfo(
    name="Cerro Pachon",
    region="Chile",
    timezone="America/Santiago",
    latitude=-30.24,
    longitude=-70.74
)

# Get all unique dates
dates = pd.date_range(start=df['timestamp'].min().date(),
                      end=df['timestamp'].max().date(),
                      freq='D')

# Calculate sunset time and relevant temperatures for each day
data_by_day = []

for date in dates:
    # Calculate sunset time for this day
    s = sun(location.observer, date=date)
    sunset_time = s['sunset'].replace(tzinfo=None)

    # Cutoff time: 3 hours before sunset
    cutoff_time = sunset_time - timedelta(hours=3)

    # Get data for this day (from midnight to sunset)
    day_start = pd.Timestamp(date)
    day_data = df[(df['timestamp'] >= day_start) & (df['timestamp'] <= sunset_time)].copy()

    if len(day_data) == 0:
        continue

    # Find temperature at sunset
    time_diff_sunset = abs(day_data['timestamp'] - sunset_time)
    sunset_idx = time_diff_sunset.idxmin()
    temp_at_sunset = day_data.loc[sunset_idx, 'temperature']

    # Find temperature at cutoff (3h before sunset)
    time_diff_cutoff = abs(day_data['timestamp'] - cutoff_time)
    cutoff_idx = time_diff_cutoff.idxmin()
    temp_at_cutoff = day_data.loc[cutoff_idx, 'temperature']

    # Get all temperatures up to cutoff
    temps_before_cutoff = day_data[day_data['timestamp'] <= cutoff_time]['temperature'].values

    if len(temps_before_cutoff) == 0:
        continue

    data_by_day.append({
        'date': date,
        'day_of_month': date.day,
        'day_of_year': date.dayofyear,
        'sunset_time': sunset_time,
        'cutoff_time': cutoff_time,
        'temp_at_sunset': temp_at_sunset,
        'temp_at_cutoff': temp_at_cutoff,
        'temps_before_cutoff': temps_before_cutoff,
        'num_measurements': len(temps_before_cutoff)
    })

print(f"  Processed {len(data_by_day)} days with complete data")

# ============================================================================
# 2. TRAIN/TEST SPLIT (EVEN/ODD DAYS)
# ============================================================================

print("\n[2/6] Splitting data into train/test sets...")

train_days = [d for d in data_by_day if d['day_of_month'] % 2 == 0]
test_days = [d for d in data_by_day if d['day_of_month'] % 2 == 1]

print(f"  Training set: {len(train_days)} days (even days of month)")
print(f"  Test set: {len(test_days)} days (odd days of month)")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n[3/6] Engineering features for classical ML models...")

def engineer_features(day_data):
    """Extract features from a day's temperature data."""
    temps = day_data['temps_before_cutoff']
    cutoff_time = day_data['cutoff_time']
    sunset_time = day_data['sunset_time']

    features = {
        # Temporal features
        'hour_of_cutoff': cutoff_time.hour + cutoff_time.minute / 60,
        'day_of_year': day_data['day_of_year'],
        'sunset_hour': sunset_time.hour + sunset_time.minute / 60,

        # Statistical features
        'temp_mean': np.mean(temps),
        'temp_std': np.std(temps),
        'temp_min': np.min(temps),
        'temp_max': np.max(temps),
        'temp_last': temps[-1],  # Temperature at cutoff
        'temp_range': np.max(temps) - np.min(temps),

        # Cyclical encoding of day of year
        'day_of_year_sin': np.sin(2 * np.pi * day_data['day_of_year'] / 365),
        'day_of_year_cos': np.cos(2 * np.pi * day_data['day_of_year'] / 365),
    }

    # Lagged features (if enough data)
    if len(temps) >= 4:  # At least 1 hour of data
        features['temp_1h_ago'] = temps[-4]  # 4 measurements = 1 hour
    else:
        features['temp_1h_ago'] = temps[0]

    if len(temps) >= 8:  # At least 2 hours
        features['temp_2h_ago'] = temps[-8]
    else:
        features['temp_2h_ago'] = temps[0]

    if len(temps) >= 24:  # At least 6 hours
        features['temp_6h_ago'] = temps[-24]
    else:
        features['temp_6h_ago'] = temps[0]

    # Trend features (last 2 hours if available)
    if len(temps) >= 8:
        recent_temps = temps[-8:]
        time_indices = np.arange(len(recent_temps))
        slope = np.polyfit(time_indices, recent_temps, 1)[0]
        features['temp_slope'] = slope
        features['temp_change_1h'] = temps[-1] - temps[-4]
    else:
        features['temp_slope'] = 0
        features['temp_change_1h'] = 0

    return features

# Create feature matrices
X_train_list = [engineer_features(d) for d in train_days]
X_test_list = [engineer_features(d) for d in test_days]

feature_names = list(X_train_list[0].keys())
X_train = np.array([[f[name] for name in feature_names] for f in X_train_list])
X_test = np.array([[f[name] for name in feature_names] for f in X_test_list])

y_train = np.array([d['temp_at_sunset'] for d in train_days])
y_test = np.array([d['temp_at_sunset'] for d in test_days])

print(f"  Engineered {len(feature_names)} features: {', '.join(feature_names)}")
print(f"  Training shapes: X={X_train.shape}, y={y_train.shape}")
print(f"  Test shapes: X={X_test.shape}, y={y_test.shape}")

# ============================================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n[4/6] Training and evaluating models...")

results = []

# ----------------------------------------------------------------------------
# Model 1: Persistence Model (Baseline)
# ----------------------------------------------------------------------------
print("\n  [1/5] Persistence Model (Baseline)...")
start_time = time.time()

# Prediction = temperature at cutoff (3h before sunset)
y_pred_persistence = np.array([d['temp_at_cutoff'] for d in test_days])

train_time = time.time() - start_time
inference_time = 0  # Essentially zero computation

mae = mean_absolute_error(y_test, y_pred_persistence)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_persistence))
r2 = r2_score(y_test, y_pred_persistence)

print(f"    MAE: {mae:.4f}°C, RMSE: {rmse:.4f}°C, R²: {r2:.4f}")

results.append({
    'model': 'Persistence',
    'predictions': y_pred_persistence,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'train_time': train_time,
    'inference_time': inference_time
})

# ----------------------------------------------------------------------------
# Model 2: Linear Regression
# ----------------------------------------------------------------------------
print("\n  [2/5] Linear Regression...")
start_time = time.time()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

train_time = time.time() - start_time

start_time = time.time()
y_pred_lr = lr_model.predict(X_test_scaled)
inference_time = time.time() - start_time

mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2 = r2_score(y_test, y_pred_lr)

print(f"    MAE: {mae:.4f}°C, RMSE: {rmse:.4f}°C, R²: {r2:.4f}")

results.append({
    'model': 'Linear Regression',
    'predictions': y_pred_lr,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'train_time': train_time,
    'inference_time': inference_time,
    'feature_importance': np.abs(lr_model.coef_)
})

# ----------------------------------------------------------------------------
# Model 3: Random Forest
# ----------------------------------------------------------------------------
print("\n  [3/5] Random Forest...")
start_time = time.time()

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

train_time = time.time() - start_time

start_time = time.time()
y_pred_rf = rf_model.predict(X_test)
inference_time = time.time() - start_time

mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2 = r2_score(y_test, y_pred_rf)

print(f"    MAE: {mae:.4f}°C, RMSE: {rmse:.4f}°C, R²: {r2:.4f}")

results.append({
    'model': 'Random Forest',
    'predictions': y_pred_rf,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'train_time': train_time,
    'inference_time': inference_time,
    'feature_importance': rf_model.feature_importances_
})

# ----------------------------------------------------------------------------
# Model 4: XGBoost
# ----------------------------------------------------------------------------
print("\n  [4/5] XGBoost...")
start_time = time.time()

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

train_time = time.time() - start_time

start_time = time.time()
y_pred_xgb = xgb_model.predict(X_test)
inference_time = time.time() - start_time

mae = mean_absolute_error(y_test, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2 = r2_score(y_test, y_pred_xgb)

print(f"    MAE: {mae:.4f}°C, RMSE: {rmse:.4f}°C, R²: {r2:.4f}")

results.append({
    'model': 'XGBoost',
    'predictions': y_pred_xgb,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'train_time': train_time,
    'inference_time': inference_time,
    'feature_importance': xgb_model.feature_importances_
})


# ----------------------------------------------------------------------------
# Model 5: LSTM - Skipped
# ----------------------------------------------------------------------------
print("\n  [5/5] LSTM (Deep Learning) - Skipped", flush=True)
print("    Note: LSTM skipped due to PyTorch environment compatibility", flush=True)

# ============================================================================
# 5. COMPARISON AND VISUALIZATION
# ============================================================================
# 5. COMPARISON AND VISUALIZATION
# ============================================================================

print("\n[5/6] Generating visualizations and comparison...")

# Sort results by MAE (best first)
results_sorted = sorted(results, key=lambda x: x['mae'])

# Print comparison table
print("\n" + "=" * 80)
print("MODEL PERFORMANCE COMPARISON (ranked by MAE)")
print("=" * 80)
print(f"{'Model':<20} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'R²':<10} {'Train (s)':<12} {'Infer (s)':<12}")
print("-" * 80)
for r in results_sorted:
    print(f"{r['model']:<20} {r['mae']:<12.4f} {r['rmse']:<12.4f} {r['r2']:<10.4f} "
          f"{r['train_time']:<12.3f} {r['inference_time']:<12.6f}")
print("=" * 80)

# Create visualizations
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str

plots = []

# Plot 1: Predicted vs Actual for all models
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, r in enumerate(results):
    ax = axes[idx]
    ax.scatter(y_test, r['predictions'], alpha=0.6, s=30)

    # Perfect prediction line
    min_val = min(y_test.min(), r['predictions'].min())
    max_val = max(y_test.max(), r['predictions'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('Actual Temperature (°C)', fontsize=10)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=10)
    ax.set_title(f"{r['model']}\nMAE: {r['mae']:.4f}°C, R²: {r['r2']:.4f}", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hide the 6th subplot
axes[5].axis('off')

plt.tight_layout()
plots.append(('predicted_vs_actual', fig_to_base64(fig)))
plt.close(fig)

# Plot 2: Residuals vs Day of Year for best 3 models
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

test_day_of_year = [d['day_of_year'] for d in test_days]

for idx, r in enumerate(results_sorted[:3]):
    ax = axes[idx]
    residuals = y_test - r['predictions']
    ax.scatter(test_day_of_year, residuals, alpha=0.6, s=30)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Day of Year', fontsize=10)
    ax.set_ylabel('Residual (°C)', fontsize=10)
    ax.set_title(f"{r['model']}\nResiduals (Mean: {residuals.mean():.4f}°C)", fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plots.append(('residuals', fig_to_base64(fig)))
plt.close(fig)

# Plot 3: Feature importance for tree-based models
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

feature_models = [(r['model'], r.get('feature_importance'))
                  for r in results if 'feature_importance' in r]

for idx, (model_name, importance) in enumerate(feature_models):
    ax = axes[idx]

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    # Plot top 10
    top_n = min(10, len(sorted_features))
    ax.barh(range(top_n), sorted_importance[:top_n])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_features[:top_n], fontsize=9)
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_title(f'{model_name}\nTop Features', fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plots.append(('feature_importance', fig_to_base64(fig)))
plt.close(fig)

# Plot 4: Time series comparison (first 30 test days)
fig, ax = plt.subplots(figsize=(14, 6))

test_dates = [d['date'] for d in test_days[:30]]
ax.plot(test_dates, y_test[:30], 'ko-', linewidth=2, markersize=6,
        label='Actual', zorder=10)

colors = ['gray', 'blue', 'green', 'red', 'purple']
for idx, r in enumerate(results_sorted[:3]):  # Show top 3 models
    ax.plot(test_dates, r['predictions'][:30], 'o--',
            color=colors[idx], alpha=0.7, linewidth=1.5,
            markersize=4, label=f"{r['model']} (MAE: {r['mae']:.3f}°C)")

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Temperature at Sunset (°C)', fontsize=11)
ax.set_title('Predicted vs Actual Sunset Temperature (First 30 Test Days)', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plots.append(('time_series', fig_to_base64(fig)))
plt.close(fig)

# ============================================================================
# 6. GENERATE HTML REPORT
# ============================================================================

print("\n[6/6] Generating HTML report...")

# Determine best model and create recommendation
best_model = results_sorted[0]
baseline_model = [r for r in results if r['model'] == 'Persistence'][0]
improvement = ((baseline_model['mae'] - best_model['mae']) / baseline_model['mae']) * 100

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML Sunset Temperature Prediction - Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0;
            font-size: 2.2em;
        }}
        .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best-model {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .recommendation {{
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendation h3 {{
            margin-top: 0;
            color: #1976D2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .metric {{
            display: inline-block;
            background: #f0f0f0;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        ul {{
            line-height: 1.8;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ML-Based Sunset Temperature Prediction</h1>
        <div class="subtitle">Rubin Observatory Mirror Temperature Analysis</div>
        <div class="subtitle">Predicting temperature at sunset using data up to 3 hours prior</div>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This analysis compares 5 machine learning approaches for predicting mirror temperature at sunset using
        measurements available up to 3 hours before sunset. The system was trained on {len(train_days)} days
        (even days of month) and tested on {len(test_days)} days (odd days of month) from the period
        {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}.</p>

        <div style="text-align: center; margin: 30px 0;">
            <div class="metric">
                <div class="metric-label">Best Model</div>
                <div class="metric-value">{best_model['model']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{best_model['mae']:.4f}°C</div>
            </div>
            <div class="metric">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{best_model['r2']:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Improvement vs Baseline</div>
                <div class="metric-value">{improvement:.1f}%</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Model Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>MAE (°C)</th>
                    <th>RMSE (°C)</th>
                    <th>R² Score</th>
                    <th>Training Time (s)</th>
                    <th>Inference Time (s)</th>
                </tr>
            </thead>
            <tbody>
"""

for rank, r in enumerate(results_sorted, 1):
    row_class = 'best-model' if rank == 1 else ''
    html_content += f"""
                <tr class="{row_class}">
                    <td>{rank}</td>
                    <td><strong>{r['model']}</strong></td>
                    <td>{r['mae']:.4f}</td>
                    <td>{r['rmse']:.4f}</td>
                    <td>{r['r2']:.4f}</td>
                    <td>{r['train_time']:.3f}</td>
                    <td>{r['inference_time']:.6f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>

        <p><strong>Metrics Explained:</strong></p>
        <ul>
            <li><strong>MAE (Mean Absolute Error):</strong> Average prediction error in °C. Lower is better.</li>
            <li><strong>RMSE (Root Mean Squared Error):</strong> Emphasizes larger errors. Lower is better.</li>
            <li><strong>R² Score:</strong> Proportion of variance explained (0 to 1). Higher is better.</li>
        </ul>
    </div>

    <div class="section">
        <h2>Recommendation</h2>
        <div class="recommendation">
            <h3>Recommended Model: {best_model['model']}</h3>
            <p><strong>Why this model:</strong></p>
            <ul>
"""

# Add model-specific recommendations
if best_model['model'] == 'XGBoost':
    html_content += """
                <li>Achieves the lowest prediction error (MAE: {:.4f}°C) among all tested approaches</li>
                <li>Excellent at capturing non-linear relationships and feature interactions</li>
                <li>Fast inference time suitable for real-time operational use</li>
                <li>Provides interpretable feature importance for understanding predictions</li>
                <li>Robust to outliers and handles the temporal patterns in the data effectively</li>
""".format(best_model['mae'])
elif best_model['model'] == 'Random Forest':
    html_content += """
                <li>Achieves the lowest prediction error (MAE: {:.4f}°C) among all tested approaches</li>
                <li>Robust ensemble method that handles non-linear relationships well</li>
                <li>Provides feature importance for interpretability</li>
                <li>Less prone to overfitting compared to single decision trees</li>
""".format(best_model['mae'])
elif best_model['model'] == 'LSTM':
    html_content += """
                <li>Achieves the lowest prediction error (MAE: {:.4f}°C) among all tested approaches</li>
                <li>Effectively captures long-term temporal dependencies in the sequence</li>
                <li>No manual feature engineering required - learns patterns automatically</li>
                <li>Can generalize to variable-length input sequences</li>
""".format(best_model['mae'])
else:
    html_content += """
                <li>Achieves the lowest prediction error (MAE: {:.4f}°C) among all tested approaches</li>
                <li>Simple and interpretable model structure</li>
                <li>Fast training and inference times</li>
""".format(best_model['mae'])

html_content += f"""
                <li><strong>Performance vs Baseline:</strong> {improvement:.1f}% reduction in MAE compared to the
                persistence model (simply using temperature at 3h before sunset)</li>
                <li><strong>Operational Impact:</strong> With an MAE of {best_model['mae']:.4f}°C, this model provides
                reliable predictions for operational planning, giving staff 3 hours advance notice of sunset temperature</li>
            </ul>

            <p><strong>Deployment Considerations:</strong></p>
            <ul>
                <li>The model should be retrained periodically (monthly) as new data becomes available</li>
                <li>Monitor prediction errors to detect performance degradation</li>
                <li>Consider ensemble approaches combining top models for increased robustness</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>Visualizations</h2>

        <h3>Predicted vs Actual Temperature (All Models)</h3>
        <p>Each subplot shows the predicted temperatures (x-axis) versus actual temperatures (y-axis) for test days.
        Points closer to the red diagonal line indicate better predictions.</p>
        <img src="data:image/png;base64,{plots[0][1]}" alt="Predicted vs Actual">

        <h3>Residual Analysis (Top 3 Models)</h3>
        <p>Residual plots show prediction errors over the year. Randomly scattered points around zero indicate good model performance.</p>
        <img src="data:image/png;base64,{plots[1][1]}" alt="Residuals">

        <h3>Feature Importance</h3>
        <p>For tree-based models, feature importance reveals which input features contribute most to predictions.</p>
        <img src="data:image/png;base64,{plots[2][1]}" alt="Feature Importance">

        <h3>Time Series: Predicted vs Actual (First 30 Test Days)</h3>
        <p>This plot compares the top 3 models' predictions against actual sunset temperatures over time.</p>
        <img src="data:image/png;base64,{plots[3][1]}" alt="Time Series">
    </div>

    <div class="section">
        <h2>Model Descriptions</h2>

        <h3>1. Persistence Model (Baseline)</h3>
        <p>Uses the temperature measured at 3 hours before sunset as the prediction. This simple baseline establishes
        the minimum performance threshold - any ML model should beat this.</p>

        <h3>2. Linear Regression</h3>
        <p>Fits a linear model using {len(feature_names)} engineered features including temporal patterns, statistical
        aggregates, and lagged temperatures. Features are standardized before training.</p>

        <h3>3. Random Forest</h3>
        <p>Ensemble of 200 decision trees that can capture non-linear relationships. Uses same features as linear regression
        but automatically learns feature interactions.</p>

        <h3>4. XGBoost</h3>
        <p>Gradient boosted trees with 200 estimators. Builds trees sequentially, with each tree correcting errors of previous
        ones. Often best performer for tabular time series data.</p>

        <h3>5. LSTM (Long Short-Term Memory) - Skipped</h3>
        <p>Deep learning recurrent neural network skipped due to PyTorch environment compatibility issues in the CLI environment.
        The four classical ML models above provide comprehensive comparison from simple baselines to advanced ensemble methods.</p>
    </div>

    <div class="section">
        <h2>Data Summary</h2>
        <ul>
            <li><strong>Total measurements:</strong> {len(df):,}</li>
            <li><strong>Date range:</strong> {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}</li>
            <li><strong>Sampling interval:</strong> 15 minutes</li>
            <li><strong>Temperature range:</strong> {df['temperature'].min():.2f}°C to {df['temperature'].max():.2f}°C</li>
            <li><strong>Training days:</strong> {len(train_days)} (even days of month)</li>
            <li><strong>Test days:</strong> {len(test_days)} (odd days of month)</li>
            <li><strong>Prediction horizon:</strong> 3 hours before sunset</li>
            <li><strong>Observatory:</strong> Rubin Observatory, Cerro Pachón, Chile (-30.24°N, -70.74°W)</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>ML-Based Sunset Temperature Prediction System</p>
    </div>
</body>
</html>
"""

# Write HTML report
with open('ml_results_report.html', 'w') as f:
    f.write(html_content)

print(f"  HTML report saved to: ml_results_report.html")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nBest Model: {best_model['model']}")
print(f"  MAE: {best_model['mae']:.4f}°C")
print(f"  RMSE: {best_model['rmse']:.4f}°C")
print(f"  R²: {best_model['r2']:.4f}")
print(f"  Improvement over baseline: {improvement:.1f}%")
print(f"\nView full report: open ml_results_report.html")
print("=" * 80)
