"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic IoT proximity sensor dataset (sensor_data.csv)
with intentional quality issues (missing values, duplicates, outliers) so the
cleaning pipeline has meaningful work to do.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SENSORS = 5
HOURS = 72                          # 3 days of readings
FREQ_SECONDS = 30                   # one reading every 30 s
MISSING_FRAC = 0.05                 # 5 % missing values
DUPLICATE_FRAC = 0.02               # 2 % duplicate rows
OUTLIER_FRAC = 0.03                 # 3 % outlier injections

# ---------------------------------------------------------------------------
# Build base timeline
# ---------------------------------------------------------------------------
start = pd.Timestamp("2025-06-01 00:00:00")
timestamps = pd.date_range(start, periods=int(HOURS * 3600 / FREQ_SECONDS),
                           freq=f"{FREQ_SECONDS}s")

rows = []
for ts in timestamps:
    for sid in range(1, NUM_SENSORS + 1):
        hour = ts.hour
        # Simulate daily pattern: closer objects during business hours
        base = 50 + 30 * np.sin(2 * np.pi * hour / 24)
        proximity = base + np.random.normal(0, 5)
        temperature = 22 + 3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.5)
        humidity = 45 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
        battery_level = max(0, min(100, 95 - (ts - start).total_seconds() / 3600 * 0.3
                                    + np.random.normal(0, 1)))
        rows.append([ts, f"SENSOR_{sid:03d}", proximity, temperature,
                      humidity, battery_level])

df = pd.DataFrame(rows, columns=["timestamp", "sensor_id", "proximity_value",
                                  "temperature", "humidity", "battery_level"])

# ---------------------------------------------------------------------------
# Inject data-quality issues
# ---------------------------------------------------------------------------
n = len(df)

# 1. Missing values (randomly null out cells)
for col in ["proximity_value", "temperature", "humidity", "battery_level"]:
    mask = np.random.rand(n) < MISSING_FRAC
    df.loc[mask, col] = np.nan

# 2. Duplicate rows
dup_idx = np.random.choice(n, size=int(n * DUPLICATE_FRAC), replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

# 3. Outliers (extreme proximity values)
outlier_idx = np.random.choice(len(df), size=int(len(df) * OUTLIER_FRAC), replace=False)
df.loc[outlier_idx, "proximity_value"] = np.random.choice([-50, 300, 500, -100],
                                                           size=len(outlier_idx))

# 4. Shuffle so data is not perfectly sorted
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.to_csv("sensor_data.csv", index=False)
print(f"✓ Generated sensor_data.csv  —  {len(df):,} rows, {df.shape[1]} columns")
