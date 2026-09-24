import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

# ==========================================
# 1. Load training and testing data
# ==========================================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Convert date_time
train["date_time"] = pd.to_datetime(train["date_time"])
test["date_time"] = pd.to_datetime(test["date_time"])

# Sort chronologically
train = train.sort_values("date_time").reset_index(drop=True)
test = test.sort_values("date_time").reset_index(drop=True)


# ==========================================
# 2. Create time features
# ==========================================

for df in [train, test]:

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


# ==========================================
# 3. Create next-hour target
# ==========================================

train["target_traffic"] = train["traffic_volume"].shift(-1)
test["target_traffic"] = test["traffic_volume"].shift(-1)

# Remove final row because next-hour value is unavailable
train = train.dropna(subset=["target_traffic"])
test = test.dropna(subset=["target_traffic"])


# ==========================================
# 4. Select AI features
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

target = "target_traffic"

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


print("========================================")
print("Q-ROUTE AI - Traffic Prediction")
print("========================================")

print("\nFeatures used by AI:")
for feature in features:
    print("-", feature)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 5. Create Random Forest model
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. Train the AI
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 7. Make predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 8. Evaluate the model
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 9. Show sample predictions
# ==========================================

results = pd.DataFrame({
    "actual_traffic": y_test.values,
    "predicted_traffic": predictions
})

print("\nSample predictions:")
print(results.head(10))


# ==========================================
# 10. Save trained model
# ==========================================

Path("models").mkdir(exist_ok=True)

model_path = "models/traffic_model.pkl"

joblib.dump(model, model_path)

print("\n========================================")
print(f"Model saved to: {model_path}")
print("========================================")