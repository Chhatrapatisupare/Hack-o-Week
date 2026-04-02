"""
sensor_cleaning_pipeline.py
============================
A production-structured, beginner-friendly data-cleaning pipeline for
Kaggle IoT proximity sensor data.

Pipeline stages
---------------
1. Load & explore raw data
2. Clean missing values  (forward-fill for time-series, mean/median for others)
3. Remove duplicates
4. Handle outliers       (IQR method — log, don't blindly remove)
5. Validate data
6. Feature engineering   (hour, day-of-week, rolling average)
7. Normalize             (MinMax scaling on proximity_value)
8. Visualize             (raw vs. cleaned proximity over time)
9. Export cleaned CSV

Dependencies: pandas, numpy, matplotlib, scikit-learn
"""

# ── Imports ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend — saves without blocking
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# ══════════════════════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV file and perform initial exploration.

    Prints: head, shape, dtypes/info summary, descriptive stats,
            and missing-value counts per column.
    """
    df = pd.read_csv(filepath)

    print("=" * 70)
    print("  RAW DATA OVERVIEW")
    print("=" * 70)

    # --- Head ---
    print("\n▸ First 5 rows:")
    print(df.head().to_string(index=False))

    # --- Shape ---
    print(f"\n▸ Shape: {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

    # --- Info (dtypes + non-null counts) ---
    print("\n▸ Column info:")
    print(f"  {'Column':<20} {'Dtype':<12} {'Non-Null':>10}")
    print("  " + "-" * 44)
    for col in df.columns:
        print(f"  {col:<20} {str(df[col].dtype):<12} {df[col].notna().sum():>10,}")

    # --- Descriptive statistics ---
    print("\n▸ Summary statistics (numeric columns):")
    print(df.describe().round(2).to_string())

    # --- Missing values ---
    missing = df.isnull().sum()
    print("\n▸ Missing values per column:")
    for col, cnt in missing.items():
        pct = cnt / len(df) * 100
        flag = " ⚠" if cnt > 0 else ""
        print(f"  {col:<20} {cnt:>6,}  ({pct:5.2f}%){flag}")

    return df


# ══════════════════════════════════════════════════════════════════════════
#  2.  CLEAN MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════
def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using strategies appropriate to each column type:

    • Time-series columns (proximity_value): forward-fill (ffill), then
      back-fill remaining edge NaNs.
    • Other numeric columns: fill with the column median (robust to skew).
    """
    df = df.copy()

    # Columns treated as time-series (sensor readings that vary over time)
    ts_cols = ["proximity_value"]

    # All other numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_numeric = [c for c in numeric_cols if c not in ts_cols]

    # --- Forward-fill time-series columns (within each sensor) ---
    for col in ts_cols:
        before = df[col].isnull().sum()
        df[col] = df.groupby("sensor_id")[col].transform(
            lambda s: s.ffill().bfill()          # ffill first, bfill residual edge NaNs
        )
        after = df[col].isnull().sum()
        print(f"  ✓ {col}: filled {before - after:,} NaNs via forward/back-fill"
              f"  (remaining: {after:,})")

    # --- Median-fill other numeric columns ---
    for col in other_numeric:
        before = df[col].isnull().sum()
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        after = df[col].isnull().sum()
        print(f"  ✓ {col}: filled {before - after:,} NaNs with median ({median_val:.2f})"
              f"  (remaining: {after:,})")

    return df


# ══════════════════════════════════════════════════════════════════════════
#  3.  REMOVE DUPLICATES
# ══════════════════════════════════════════════════════════════════════════
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and report how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"  ✓ Removed {removed:,} duplicate rows  "
          f"({before:,} → {len(df):,})")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  4.  HANDLE OUTLIERS (IQR METHOD)
# ══════════════════════════════════════════════════════════════════════════
def handle_outliers(df: pd.DataFrame,
                    columns: list[str] | None = None,
                    iqr_factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using the IQR method and **cap** (winsorize) them
    rather than dropping rows outright.  This preserves time-series
    continuity while limiting extreme values.

    For each column:
      • Compute Q1, Q3, IQR
      • Lower fence = Q1 − iqr_factor × IQR
      • Upper fence = Q3 + iqr_factor × IQR
      • Values beyond fences → clipped to the fence value
      • Number of affected rows is logged
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n  IQR factor: {iqr_factor}")
    print(f"  {'Column':<20} {'Q1':>8} {'Q3':>8} {'IQR':>8} "
          f"{'Lower':>8} {'Upper':>8} {'Outliers':>9}")
    print("  " + "-" * 73)

    total_affected = 0
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = outlier_mask.sum()
        total_affected += n_outliers

        # Winsorize: clip to fence boundaries
        df[col] = df[col].clip(lower=lower, upper=upper)

        print(f"  {col:<20} {q1:>8.2f} {q3:>8.2f} {iqr:>8.2f} "
              f"{lower:>8.2f} {upper:>8.2f} {n_outliers:>9,}")

    print(f"\n  ✓ Total outlier cells capped: {total_affected:,}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  5.  VALIDATE DATA
# ══════════════════════════════════════════════════════════════════════════
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run sanity checks on the cleaned data:
      • Ensure proximity_value ≥ 0  (negative readings are invalid)
      • Ensure battery_level ∈ [0, 100]
      • Report remaining NaN counts
    """
    print("\n" + "=" * 70)
    print("  DATA VALIDATION")
    print("=" * 70)

    # --- Check for negative proximity ---
    neg_prox = (df["proximity_value"] < 0).sum()
    if neg_prox > 0:
        print(f"  ⚠ Found {neg_prox:,} negative proximity values → setting to 0")
        df.loc[df["proximity_value"] < 0, "proximity_value"] = 0.0
    else:
        print("  ✓ No negative proximity values")

    # --- Check battery bounds ---
    if "battery_level" in df.columns:
        bad_bat = ((df["battery_level"] < 0) | (df["battery_level"] > 100)).sum()
        if bad_bat > 0:
            print(f"  ⚠ Found {bad_bat:,} out-of-range battery values → clipping to [0, 100]")
            df["battery_level"] = df["battery_level"].clip(0, 100)
        else:
            print("  ✓ Battery levels within [0, 100]")

    # --- Remaining NaNs ---
    remaining = df.isnull().sum().sum()
    if remaining > 0:
        print(f"  ⚠ {remaining:,} NaN cells still remain")
    else:
        print("  ✓ No remaining NaN values")

    # --- Clean summary ---
    print("\n▸ Clean dataset summary:")
    print(df.describe().round(2).to_string())

    return df


# ══════════════════════════════════════════════════════════════════════════
#  6.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
def feature_engineering(df: pd.DataFrame,
                        rolling_window: int = 10) -> pd.DataFrame:
    """
    Derive useful features from the cleaned dataset:

    • hour        — hour of day  (0-23)
    • day_of_week — day name     (Monday … Sunday)
    • prox_rolling_avg — rolling mean of proximity_value per sensor
                         to smooth high-frequency noise.
    """
    df = df.copy()

    # Hour of day
    df["hour"] = df["timestamp"].dt.hour

    # Day of week (as readable name)
    df["day_of_week"] = df["timestamp"].dt.day_name()

    # Rolling average per sensor (window = rolling_window readings)
    df["prox_rolling_avg"] = (
        df.groupby("sensor_id")["proximity_value"]
          .transform(lambda s: s.rolling(window=rolling_window,
                                          min_periods=1).mean())
    )

    print(f"  ✓ Added 'hour', 'day_of_week', 'prox_rolling_avg' "
          f"(window={rolling_window})")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  7.  NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════
def normalize(df: pd.DataFrame,
              columns: list[str] | None = None) -> pd.DataFrame:
    """
    Apply MinMax scaling to specified columns (default: proximity_value).
    Adds new column(s) with the suffix '_norm'.
    """
    df = df.copy()
    if columns is None:
        columns = ["proximity_value"]

    scaler = MinMaxScaler()
    for col in columns:
        norm_col = f"{col}_norm"
        df[norm_col] = scaler.fit_transform(df[[col]])
        print(f"  ✓ MinMax scaled '{col}' → '{norm_col}'  "
              f"[{df[col].min():.2f}, {df[col].max():.2f}] → [0, 1]")

    return df


# ══════════════════════════════════════════════════════════════════════════
#  8.  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
def visualize(raw_df: pd.DataFrame,
              clean_df: pd.DataFrame,
              sensor_id: str | None = None) -> None:
    """
    Plot raw vs. cleaned proximity_value over time for a single sensor.
    If sensor_id is None, the first sensor found in the data is used.
    """
    if sensor_id is None:
        sensor_id = raw_df["sensor_id"].unique()[0]

    raw_s = (raw_df[raw_df["sensor_id"] == sensor_id]
             .sort_values("timestamp"))
    cln_s = (clean_df[clean_df["sensor_id"] == sensor_id]
             .sort_values("timestamp"))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # --- Raw ---
    axes[0].plot(raw_s["timestamp"], raw_s["proximity_value"],
                 color="#e74c3c", alpha=0.7, linewidth=0.6, label="Raw")
    axes[0].set_title(f"Raw Proximity — {sensor_id}", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Proximity Value")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # --- Cleaned + rolling avg ---
    axes[1].plot(cln_s["timestamp"], cln_s["proximity_value"],
                 color="#2ecc71", alpha=0.7, linewidth=0.6, label="Cleaned")
    if "prox_rolling_avg" in cln_s.columns:
        axes[1].plot(cln_s["timestamp"], cln_s["prox_rolling_avg"],
                     color="#3498db", linewidth=1.4, label="Rolling Avg")
    axes[1].set_title(f"Cleaned Proximity — {sensor_id}", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Proximity Value")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("raw_vs_cleaned.png", dpi=150)
    plt.close(fig)
    print("  ✓ Saved plot → raw_vs_cleaned.png")


# ══════════════════════════════════════════════════════════════════════════
#  9.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    INPUT_FILE = "sensor_data.csv"
    OUTPUT_FILE = "cleaned_sensor_data.csv"

    # ── Step 1 · Load & explore ──────────────────────────────────────────
    raw_df = load_data(INPUT_FILE)
    initial_rows = len(raw_df)

    # Keep a copy of raw data for the before/after comparison plot
    raw_copy = raw_df.copy()

    # ── Step 2 · Convert timestamp & sort ────────────────────────────────
    print("\n" + "=" * 70)
    print("  TIMESTAMP CONVERSION & SORTING")
    print("=" * 70)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    print("  ✓ Converted 'timestamp' to datetime and sorted by sensor + time")

    # Also fix raw_copy for plotting
    raw_copy["timestamp"] = pd.to_datetime(raw_copy["timestamp"])

    # ── Step 3 · Handle missing values ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  MISSING VALUE TREATMENT")
    print("=" * 70)
    df = clean_missing_values(raw_df)

    # ── Step 4 · Remove duplicates ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DUPLICATE REMOVAL")
    print("=" * 70)
    df = remove_duplicates(df)

    # ── Step 5 · Handle outliers ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OUTLIER HANDLING (IQR)")
    print("=" * 70)
    df = handle_outliers(df,
                         columns=["proximity_value", "temperature",
                                  "humidity", "battery_level"])

    # ── Step 6 · Validate ────────────────────────────────────────────────
    df = validate_data(df)

    # ── Step 7 · Feature engineering ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING")
    print("=" * 70)
    df = feature_engineering(df, rolling_window=10)

    # ── Step 8 · Normalization ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  NORMALIZATION")
    print("=" * 70)
    df = normalize(df, columns=["proximity_value"])

    # ── Step 9 · Export ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXPORT")
    print("=" * 70)
    df.to_csv(OUTPUT_FILE, index=False)
    final_rows = len(df)
    print(f"  ✓ Saved cleaned dataset → {OUTPUT_FILE}")
    print(f"\n  Before: {initial_rows:,} rows")
    print(f"  After:  {final_rows:,} rows")
    print(f"  Removed: {initial_rows - final_rows:,} rows "
          f"({(initial_rows - final_rows) / initial_rows * 100:.1f}%)")

    # ── Step 10 · Visualize ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VISUALIZATION")
    print("=" * 70)
    visualize(raw_copy, df)

    print("\n" + "=" * 70)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
