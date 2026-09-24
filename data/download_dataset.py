from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Fetch UCI Metro Interstate Traffic Volume dataset
dataset = fetch_ucirepo(id=492)

# Get features and target
features = dataset.data.features
target = dataset.data.targets

# Combine everything
df = pd.concat([features, target], axis=1)

# Make sure data folder exists
Path("data").mkdir(exist_ok=True)

# Save dataset
output_path = "data/traffic_raw.csv"
df.to_csv(output_path, index=False)

print("Dataset downloaded successfully!")
print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())