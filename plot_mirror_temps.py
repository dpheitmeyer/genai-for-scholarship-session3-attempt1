#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "astral",
#     "numpy",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astral import LocationInfo
from astral.sun import sun
from datetime import timedelta

# Read the CSV file
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])

# Filter to the most recent week
last_timestamp = df['timestamp'].max()
one_week_ago = last_timestamp - timedelta(days=7)
df_week = df[df['timestamp'] >= one_week_ago].copy()

# Calculate sunset times for each day
location = LocationInfo(
    name="Cerro Pachon",
    region="Chile",
    timezone="America/Santiago",
    latitude=-30.24,
    longitude=-70.74
)

# Get unique dates in the week
dates = pd.date_range(start=one_week_ago.date(), end=last_timestamp.date(), freq='D')
sunset_times = []
sunset_temps = []

for date in dates:
    s = sun(location.observer, date=date)
    sunset_time = s['sunset']

    # Convert to timezone-naive to match CSV timestamps
    sunset_time_naive = sunset_time.replace(tzinfo=None)

    # Find temperature closest to sunset time
    time_diff = abs(df_week['timestamp'] - sunset_time_naive)
    closest_idx = time_diff.idxmin()

    if closest_idx in df_week.index:
        sunset_times.append(df_week.loc[closest_idx, 'timestamp'])
        sunset_temps.append(df_week.loc[closest_idx, 'temperature'])

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df_week['timestamp'], df_week['temperature'], linewidth=0.5)
plt.scatter(sunset_times, sunset_temps, color='red', s=100, zorder=5, label='Sunset')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Rubin Observatory Mirror Temperature - Most Recent Week')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Create a second figure for the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['temperature'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.title('Rubin Observatory Mirror Temperature Distribution')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Fourier transform analysis
# Calculate sampling interval (time between measurements)
time_diffs = df['timestamp'].diff()
dt = time_diffs.median().total_seconds()  # seconds
sampling_rate = 1.0 / dt  # Hz

# Perform FFT on temperature data
temps = df['temperature'].values
n = len(temps)
fft_vals = np.fft.fft(temps)
freqs = np.fft.fftfreq(n, d=dt)

# Calculate power spectrum (only positive frequencies)
power = np.abs(fft_vals) ** 2
positive_freqs = freqs[:n//2]
positive_power = power[:n//2]

# Convert frequencies to periods in days
# Exclude zero frequency (DC component)
nonzero_mask = positive_freqs > 0
periods_seconds = 1.0 / positive_freqs[nonzero_mask]
periods_days = periods_seconds / (24 * 3600)
power_nonzero = positive_power[nonzero_mask]

# Find dominant periods (top 5 peaks)
# Sort by power and get the indices
sorted_indices = np.argsort(power_nonzero)[::-1]
top_n = 5
dominant_periods = periods_days[sorted_indices[:top_n]]
dominant_powers = power_nonzero[sorted_indices[:top_n]]

print("\nDominant periods in the temperature time series:")
for i, (period, power_val) in enumerate(zip(dominant_periods, dominant_powers)):
    print(f"{i+1}. Period: {period:.2f} days (Power: {power_val:.2e})")

# Create a third figure for the power spectrum
plt.figure(figsize=(12, 6))
plt.loglog(periods_days, power_nonzero)
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.title('Power Spectrum of Mirror Temperature Time Series')
plt.grid(True, alpha=0.3, which='both')
plt.xlim([0.1, periods_days.max()])

# Mark dominant periods
for period in dominant_periods[:3]:  # Mark top 3
    plt.axvline(period, color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()

plt.show()
