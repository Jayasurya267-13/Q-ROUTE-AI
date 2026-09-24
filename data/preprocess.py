import pandas as pd
from pathlib import Path

INPUT_FILE = "data/traffic_raw.csv"
OUTPUT_FILE = "data/traffic_processed.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["date_time"] = pd.to_datetime(df["date_time"])

# Sort chronologically
df = df.sort_values("date_time").reset_index(drop=True)

# Create time features
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["year"] = df["date_time"].dt.year
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Create congestion levels using traffic-volume quantiles
low_threshold = df["traffic_volume"].quantile(0.33)
high_threshold = df["traffic_volume"].quantile(0.66)

def classify_congestion(volume):
    if volume <= low_threshold:
        return "LOW"
    elif volume <= high_threshold:
        return "MEDIUM"
    else:
        return "HIGH"

df["congestion_level"] = df["traffic_volume"].apply(
    classify_congestion
)

# Save processed data
Path("data").mkdir(exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Preprocessing completed.")
print(f"Rows: {len(df)}")
print(f"Low threshold: {low_threshold:.2f}")
print(f"High threshold: {high_threshold:.2f}")

print("\nCongestion distribution:")
print(df["congestion_level"].value_counts())

print("\nProcessed columns:")
print(df.columns.tolist())