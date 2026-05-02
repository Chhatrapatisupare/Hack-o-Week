"""
sensor_cleaning_pipeline.py

Data cleaning pipeline for IoT proximity sensor data.

Steps:
1. Load data
2. Handle missing values
3. Remove duplicates
4. Handle outliers (IQR)
5. Validate data
6. Feature engineering
7. Normalize
8. Visualize
9. Export
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and print basic overview."""
    df = pd.read_csv(filepath)

    print("\n[INFO] Raw Data Overview")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(df.describe().round(2))

    missing = df.isnull().sum()
    print("\nMissing values:")
    print(missing[missing > 0])

    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using time-series and median strategies."""
    df = df.copy()

    ts_cols = ["proximity_value"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_numeric = [c for c in numeric_cols if c not in ts_cols]

    for col in ts_cols:
        df[col] = df.groupby("sensor_id")[col].transform(lambda s: s.ffill().bfill())

    for col in other_numeric:
        df[col] = df[col].fillna(df[col].median())

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[INFO] Removed {before - len(df)} duplicate rows")
    return df


def handle_outliers(df: pd.DataFrame,
                    columns: list[str] | None = None,
                    iqr_factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers using IQR method."""
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run sanity checks."""
    if (df["proximity_value"] < 0).any():
        df.loc[df["proximity_value"] < 0, "proximity_value"] = 0

    if "battery_level" in df.columns:
        df["battery_level"] = df["battery_level"].clip(0, 100)

    return df


def feature_engineering(df: pd.DataFrame,
                        rolling_window: int = 10) -> pd.DataFrame:
    """Create derived features."""
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()

    df["prox_rolling_avg"] = (
        df.groupby("sensor_id")["proximity_value"]
        .transform(lambda s: s.rolling(window=rolling_window, min_periods=1).mean())
    )

    return df


def normalize(df: pd.DataFrame,
              columns: list[str] | None = None) -> pd.DataFrame:
    """Apply MinMax scaling."""
    df = df.copy()

    if columns is None:
        columns = ["proximity_value"]

    scaler = MinMaxScaler()
    for col in columns:
        df[f"{col}_norm"] = scaler.fit_transform(df[[col]])

    return df


def visualize(raw_df: pd.DataFrame,
              clean_df: pd.DataFrame,
              sensor_id: str | None = None) -> None:
    """Plot raw vs cleaned data."""
    if sensor_id is None:
        sensor_id = raw_df["sensor_id"].iloc[0]

    raw = raw_df[raw_df["sensor_id"] == sensor_id].sort_values("timestamp")
    clean = clean_df[clean_df["sensor_id"] == sensor_id].sort_values("timestamp")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(raw["timestamp"], raw["proximity_value"], label="Raw")
    axes[0].set_title(f"Raw - {sensor_id}")

    axes[1].plot(clean["timestamp"], clean["proximity_value"], label="Cleaned")
    if "prox_rolling_avg" in clean.columns:
        axes[1].plot(clean["timestamp"], clean["prox_rolling_avg"], label="Rolling Avg")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("raw_vs_cleaned.png")
    plt.close()


def main() -> None:
    INPUT_FILE = "sensor_data.csv"
    OUTPUT_FILE = "cleaned_sensor_data.csv"

    raw_df = load_data(INPUT_FILE)
    raw_copy = raw_df.copy()

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(["sensor_id", "timestamp"])

    raw_copy["timestamp"] = pd.to_datetime(raw_copy["timestamp"])

    df = clean_missing_values(raw_df)
    df = remove_duplicates(df)
    df = handle_outliers(df, ["proximity_value", "temperature", "humidity", "battery_level"])
    df = validate_data(df)
    df = feature_engineering(df)
    df = normalize(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved cleaned data → {OUTPUT_FILE}")

    visualize(raw_copy, df)


if __name__ == "__main__":
    main()
