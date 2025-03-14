import pandas as pd
import joblib

class DonationModelPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        self.model = joblib.load(self.model_path)
    
    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        return prediction[0]

# Example Usage
if __name__ == "__main__":
    predictor = DonationModelPredictor('Models/trained_donation_model.pkl')
    predictor.load_model()
    
    # Example input data (values should match feature order used during training)
    input_data = {
        "Routes Completed": 5,
        "Time to Complete (min)": 120,
        "Doors in Route": 25,
        "AvgDoorsPerRoute": 10,
        "Total Volunteers": 5,
        "# of Adult Volunteers": 3,
        "# of Youth Volunteers": 2,
        "VolunteersPerDoor": 3,
        "TimePerVolunteer": 5
    }
    
    prediction = predictor.predict(input_data)
    print(f"Predicted Total Donation Volume: {prediction}")
