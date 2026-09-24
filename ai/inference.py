import pandas as pd
import joblib


class TrafficPredictor:

    def __init__(
        self,
        model_path="models/traffic_model.pkl",
        training_data_path="data/train.csv"
    ):

        # Load trained AI model
        self.model = joblib.load(model_path)

        # Load training data for congestion thresholds
        train = pd.read_csv(training_data_path)

        self.low_threshold = train["traffic_volume"].quantile(0.33)
        self.high_threshold = train["traffic_volume"].quantile(0.66)

        # Features used during model training
        self.features = [
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

    def predict(
        self,
        traffic_volume,
        temp,
        rain_1h,
        snow_1h,
        clouds_all,
        hour,
        day_of_week,
        month
    ):

        # Determine whether it is a weekend
        is_weekend = 1 if day_of_week >= 5 else 0

        # Create input dataframe
        input_data = pd.DataFrame([{
            "traffic_volume": traffic_volume,
            "temp": temp,
            "rain_1h": rain_1h,
            "snow_1h": snow_1h,
            "clouds_all": clouds_all,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend
        }])

        # Predict next-hour traffic
        predicted_traffic = self.model.predict(
            input_data[self.features]
        )[0]

        # Classify congestion
        if predicted_traffic <= self.low_threshold:
            congestion = "LOW"

        elif predicted_traffic <= self.high_threshold:
            congestion = "MEDIUM"

        else:
            congestion = "HIGH"

        return {
            "predicted_traffic": float(predicted_traffic),
            "congestion": congestion
        }


# ==========================================
# Test the inference module
# ==========================================

if __name__ == "__main__":

    predictor = TrafficPredictor()

    result = predictor.predict(
        traffic_volume=2500,
        temp=288.5,
        rain_1h=0.0,
        snow_1h=0.0,
        clouds_all=40,
        hour=8,
        day_of_week=2,
        month=10
    )

    print("\n========================================")
    print("Q-ROUTE AI - INFERENCE TEST")
    print("========================================")

    print(f"Predicted traffic : {result['predicted_traffic']:.2f}")
    print(f"Congestion level  : {result['congestion']}")