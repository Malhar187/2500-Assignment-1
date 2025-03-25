import pandas as pd
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DonationModelPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        logger.info(f"Initialized DonationModelPredictor with model path: {self.model_path}")
    
    def load_model(self):
        """Loads the pre-trained model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def predict(self, input_data):
        """Makes a prediction using the loaded model."""
        if self.model is None:
            logger.error("Model is not loaded. Call load_model() first.")
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        try:
            input_df = pd.DataFrame([input_data])
            prediction = self.model.predict(input_df)
            logger.info(f"Prediction made successfully: {prediction[0]}")
            return prediction[0]
        except Exception as e:
            logger.error(f"Error occurred while making prediction: {e}")
            raise

# Example Usage
if __name__ == "__main__":
    try:
        logger.info("Starting prediction process.")
        
        # Initialize the predictor
        predictor = DonationModelPredictor('Models/trained_donation_model.pkl')
        
        # Load the pre-trained model
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
        
        # Make the prediction
        prediction = predictor.predict(input_data)
        
        # Output the result
        logger.info(f"Predicted Total Donation Volume: {prediction}")
        print(f"Predicted Total Donation Volume: {prediction}")
        
    except Exception as e:
        logger.error(f"An error occurred during the prediction process: {e}")