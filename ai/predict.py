class TrafficPredictor:

    def __init__(self, model=None):
        self.model = model

    def predict(self, traffic_data):
        """
        Predict future traffic congestion.
        """
        if self.model is None:
            raise ValueError("Traffic prediction model is not loaded.")

        return self.model.predict(traffic_data)