#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "xgboost",
#     "astral",
# ]
# ///

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astral import LocationInfo
from astral.sun import sun
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

print("Generating ML comparison PNG visualizations...")

# Load and process data (same as main script)
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

location = LocationInfo(
    name="Cerro Pachon",
    region="Chile",
    timezone="America/Santiago",
    latitude=-30.24,
    longitude=-70.74
)

dates = pd.date_range(start=df['timestamp'].min().date(),
                      end=df['timestamp'].max().date(),
                      freq='D')

data_by_day = []
for date in dates:
    s = sun(location.observer, date=date)
    sunset_time = s['sunset'].replace(tzinfo=None)
    cutoff_time = sunset_time - timedelta(hours=3)

    day_start = pd.Timestamp(date)
    day_data = df[(df['timestamp'] >= day_start) & (df['timestamp'] <= sunset_time)].copy()

    if len(day_data) == 0:
        continue

    time_diff_sunset = abs(day_data['timestamp'] - sunset_time)
    sunset_idx = time_diff_sunset.idxmin()
    temp_at_sunset = day_data.loc[sunset_idx, 'temperature']

    time_diff_cutoff = abs(day_data['timestamp'] - cutoff_time)
    cutoff_idx = time_diff_cutoff.idxmin()
    temp_at_cutoff = day_data.loc[cutoff_idx, 'temperature']

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
    })

train_days = [d for d in data_by_day if d['day_of_month'] % 2 == 0]
test_days = [d for d in data_by_day if d['day_of_month'] % 2 == 1]

def engineer_features(day_data):
    temps = day_data['temps_before_cutoff']
    cutoff_time = day_data['cutoff_time']
    sunset_time = day_data['sunset_time']

    features = {
        'hour_of_cutoff': cutoff_time.hour + cutoff_time.minute / 60,
        'day_of_year': day_data['day_of_year'],
        'sunset_hour': sunset_time.hour + sunset_time.minute / 60,
        'temp_mean': np.mean(temps),
        'temp_std': np.std(temps),
        'temp_min': np.min(temps),
        'temp_max': np.max(temps),
        'temp_last': temps[-1],
        'temp_range': np.max(temps) - np.min(temps),
        'day_of_year_sin': np.sin(2 * np.pi * day_data['day_of_year'] / 365),
        'day_of_year_cos': np.cos(2 * np.pi * day_data['day_of_year'] / 365),
    }

    if len(temps) >= 4:
        features['temp_1h_ago'] = temps[-4]
    else:
        features['temp_1h_ago'] = temps[0]

    if len(temps) >= 8:
        features['temp_2h_ago'] = temps[-8]
    else:
        features['temp_2h_ago'] = temps[0]

    if len(temps) >= 24:
        features['temp_6h_ago'] = temps[-24]
    else:
        features['temp_6h_ago'] = temps[0]

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

X_train_list = [engineer_features(d) for d in train_days]
X_test_list = [engineer_features(d) for d in test_days]

feature_names = list(X_train_list[0].keys())
X_train = np.array([[f[name] for name in feature_names] for f in X_train_list])
X_test = np.array([[f[name] for name in feature_names] for f in X_test_list])

y_train = np.array([d['temp_at_sunset'] for d in train_days])
y_test = np.array([d['temp_at_sunset'] for d in test_days])

# Train all models
results = []

# Persistence
y_pred_persistence = np.array([d['temp_at_cutoff'] for d in test_days])
results.append({
    'model': 'Persistence',
    'predictions': y_pred_persistence,
    'mae': mean_absolute_error(y_test, y_pred_persistence),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_persistence)),
    'r2': r2_score(y_test, y_pred_persistence)
})

# Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
results.append({
    'model': 'Linear Regression',
    'predictions': y_pred_lr,
    'mae': mean_absolute_error(y_test, y_pred_lr),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'r2': r2_score(y_test, y_pred_lr)
})

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
results.append({
    'model': 'Random Forest',
    'predictions': y_pred_rf,
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'r2': r2_score(y_test, y_pred_rf)
})

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.1,
                             n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results.append({
    'model': 'XGBoost',
    'predictions': y_pred_xgb,
    'mae': mean_absolute_error(y_test, y_pred_xgb),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'r2': r2_score(y_test, y_pred_xgb)
})

# Add placeholder for LSTM (skipped)
results.append({
    'model': 'LSTM (Skipped)',
    'predictions': None,
    'mae': None,
    'rmse': None,
    'r2': None
})

results_sorted = sorted([r for r in results if r['mae'] is not None], key=lambda x: x['mae'])

print("Generating ml_comparison_v2.png...")

# Figure 1: Model Comparison - Predicted vs Actual
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, r in enumerate(results_sorted):
    ax = axes[idx]
    ax.scatter(y_test, r['predictions'], alpha=0.6, s=40, edgecolors='black', linewidth=0.5)

    min_val = min(y_test.min(), r['predictions'].min())
    max_val = max(y_test.max(), r['predictions'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')

    ax.set_xlabel('Actual Temperature at Sunset (°C)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title(f"{r['model']}\nMAE: {r['mae']:.3f}°C | R²: {r['r2']:.3f}",
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')

plt.suptitle('ML Model Comparison: Predicted vs Actual Sunset Temperature',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('ml_comparison_v2.png', dpi=150, bbox_inches='tight')
plt.close()

print("Generating ml_residuals_v2.png...")

# Figure 2: Residual Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

test_day_of_year = [d['day_of_year'] for d in test_days]

for idx, r in enumerate(results_sorted):
    ax = axes[idx]
    residuals = y_test - r['predictions']

    ax.scatter(test_day_of_year, residuals, alpha=0.6, s=40,
               c=residuals, cmap='RdBu_r', edgecolors='black', linewidth=0.5,
               vmin=-4, vmax=4)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.axhline(y=residuals.std(), color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±1σ ({residuals.std():.2f}°C)')
    ax.axhline(y=-residuals.std(), color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residual (Actual - Predicted) °C', fontsize=11, fontweight='bold')
    ax.set_title(f"{r['model']}\nMean: {residuals.mean():.3f}°C | Std: {residuals.std():.3f}°C",
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-4, 4)

plt.suptitle('Residual Analysis: Model Prediction Errors Over Time',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('ml_residuals_v2.png', dpi=150, bbox_inches='tight')
plt.close()

print("PNG files generated successfully!")
print("  - ml_comparison_v2.png")
print("  - ml_residuals_v2.png")
