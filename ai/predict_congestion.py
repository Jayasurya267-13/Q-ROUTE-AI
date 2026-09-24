import pandas as pd
import joblib

# ==========================================
# 1. Load trained model
# ==========================================

MODEL_FILE = "models/traffic_model.pkl"

model = joblib.load(MODEL_FILE)

print("Traffic prediction model loaded successfully.")


# ==========================================
# 2. Load training and testing data
# ==========================================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["date_time"] = pd.to_datetime(train["date_time"])
test["date_time"] = pd.to_datetime(test["date_time"])


# ==========================================
# 3. Create time features
# ==========================================

for df in [train, test]:

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


# ==========================================
# 4. Select same features used during training
# ==========================================

features = [
    "traffic_volume",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "month",
    "is_weekend"
]

X_test = test[features]


# ==========================================
# 5. Predict future traffic
# ==========================================

predictions = model.predict(X_test)

test["predicted_traffic"] = predictions


# ==========================================
# 6. Calculate congestion thresholds
# ==========================================

low_threshold = train["traffic_volume"].quantile(0.33)
high_threshold = train["traffic_volume"].quantile(0.66)

print("\nCongestion thresholds:")
print(f"LOW    : <= {low_threshold:.2f}")
print(f"MEDIUM : {low_threshold:.2f} - {high_threshold:.2f}")
print(f"HIGH   : > {high_threshold:.2f}")


# ==========================================
# 7. Classify predicted congestion
# ==========================================

def classify_congestion(volume):

    if volume <= low_threshold:
        return "LOW"

    elif volume <= high_threshold:
        return "MEDIUM"

    else:
        return "HIGH"


test["predicted_congestion"] = test["predicted_traffic"].apply(
    classify_congestion
)


# ==========================================
# 8. Save predictions
# ==========================================

output_columns = [
    "date_time",
    "traffic_volume",
    "predicted_traffic",
    "predicted_congestion"
]

results = test[output_columns]

results.to_csv(
    "data/traffic_predictions.csv",
    index=False
)


# ==========================================
# 9. Display results
# ==========================================

print("\n========================================")
print("TRAFFIC PREDICTION RESULTS")
print("========================================")

print(results.head(10))

print("\nPredicted congestion distribution:")
print(results["predicted_congestion"].value_counts())

print("\nPrediction file saved:")
print("data/traffic_predictions.csv")