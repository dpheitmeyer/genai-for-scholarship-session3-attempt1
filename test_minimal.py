#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "astral",
# ]
# ///

import sys
import pandas as pd
import numpy as np
from astral import LocationInfo
from astral.sun import sun

print("Starting test...", flush=True)

# Load data
df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])
print(f"Loaded {len(df)} measurements", flush=True)

# Test sunset calculation
location = LocationInfo(
    name="Cerro Pachon",
    region="Chile",
    timezone="America/Santiago",
    latitude=-30.24,
    longitude=-70.74
)

date = df['timestamp'].min().date()
s = sun(location.observer, date=date)
sunset_time = s['sunset']
print(f"Sunset time for {date}: {sunset_time}", flush=True)

print("Test completed successfully!", flush=True)
