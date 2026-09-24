import pandas as pd

INPUT_FILE = "data/traffic_processed.csv"

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

df = pd.read_csv(INPUT_FILE)

df["date_time"] = pd.to_datetime(df["date_time"])

# Sort by time
df = df.sort_values("date_time")

# Use the final year as the test period
test_start = df["date_time"].max() - pd.DateOffset(years=1)

train = df[df["date_time"] < test_start].copy()
test = df[df["date_time"] >= test_start].copy()

train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)

print("Time-based split completed.")

print("\nTraining:")
print(train.shape)
print(train["date_time"].min())
print(train["date_time"].max())

print("\nTesting:")
print(test.shape)
print(test["date_time"].min())
print(test["date_time"].max())