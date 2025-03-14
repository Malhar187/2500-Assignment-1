from flask import Flask, request, jsonify
import joblib
import pandas as pd
from predict import DonationModelPredictor

app = Flask(__name__)

model_v1_path = 'Models/trained_donation_model.pkl'
model_v2_path = 'Models/newly_trained_model.pkl'

predictor_v1 = DonationModelPredictor(model_v1_path)
predictor_v1.load_model()

predictor_v2 = DonationModelPredictor(model_v2_path)
predictor_v2.load_model()

@app.route('/food_drive_home', methods=['GET'])
def home():
    app_info = {
        "name": "Food Drive API",
        "description": "This API takes in different inputs and gives donation predictions.",
        "version": "v1.0",
        "endpoints": {
            "/food_drive_home": "Home Page",
            "/health_status": "Check API's health",
            "/predict/v1": {
                "description": "Predicts donation bags using model_v1",
                "method": "POST",
                "expected_input": [
                    "Routes Completed", "Time to Complete (min)", "Doors in Route", 
                    "AvgDoorsPerRoute", "Total Volunteers", "# of Adult Volunteers", 
                    "# of Youth Volunteers", "VolunteersPerDoor", "TimePerVolunteer"
                ]
            },
            "/predict/v2": {
                "description": "Predicts donation bags using model_v2",
                "method": "POST",
                "expected_input": [
                    "AvgDoorsPerRoute", "Time to Complete (min)", "Doors in Route", 
                    "Routes Completed", "# of Adult Volunteers"
                ]
            }
        }
    }
    
    return jsonify(app_info)

@app.route('/health_status', methods=['GET'])
def health_status():
    return jsonify({"status": "API is up and running!", "message": "Healthy"}), 200

@app.route("/predict/v1", methods=["POST"])
def predict_v1():
    try:
        # Step 1: Get input JSON from the request
        input_data = request.get_json()
        print(f"Received input data: {input_data}")  # Debugging: print received data

        # Step 2: Ensure input_data is a list of dictionaries
        if isinstance(input_data, dict):
            input_data = [input_data]  # If it's a single dictionary, convert to list

        print(f"Data after ensuring list format: {input_data}")  # Debugging: print the list
        
        # Step 3: Check that input_data is a list of dictionaries and convert it to a DataFrame
        if isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            input_df = pd.DataFrame(input_data)
        else:
            return jsonify({"error": "Invalid input format. Send a list of dictionaries or a single dictionary."}), 400

        # Step 4: Ensure all required features are present in the input
        expected_features = [
            "Routes Completed", "Time to Complete (min)", "Doors in Route", "AvgDoorsPerRoute",
            "Total Volunteers", "# of Adult Volunteers", "# of Youth Volunteers",
            "VolunteersPerDoor", "TimePerVolunteer"
        ]
        
        # Ensure that all expected features are in the DataFrame
        missing_cols = [col for col in expected_features if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns: {missing_cols}"}), 400

        # Step 5: Extract the data for prediction (assuming input_df has the correct columns)
        input_data_for_prediction = input_df[expected_features].iloc[0].to_dict()  # Extract data as dictionary
        print(f"Data extracted for prediction: {input_data_for_prediction}")  # Debugging

        # Step 6: Use the predictor to get the prediction
        prediction = predictor_v1.predict(input_data_for_prediction)

        return jsonify({"Predicted Donation Bags with model_v1": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

     
@app.route("/predict/v2", methods=["POST"])
def predict_v2():
    try:
        # Step 1: Get input JSON from the request
        input_data = request.get_json()
        print(f"Received input data: {input_data}")  # Debugging: print received data

        # Step 2: Ensure input_data is a list of dictionaries
        if isinstance(input_data, dict):
            input_data = [input_data]  # If it's a single dictionary, convert to list

        print(f"Data after ensuring list format: {input_data}")  # Debugging: print the list
        
        # Step 3: Check that input_data is a list of dictionaries and convert it to a DataFrame
        if isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            input_df = pd.DataFrame(input_data)
        else:
            return jsonify({"error": "Invalid input format. Send a list of dictionaries or a single dictionary."}), 400

        # Step 4: Ensure all required features are present in the input for model_v2
        expected_features_v2 = [
            "AvgDoorsPerRoute", "Time to Complete (min)", "Doors in Route", "Routes Completed", "# of Adult Volunteers"
        ]
        
        # Ensure that all expected features are in the DataFrame
        missing_cols = [col for col in expected_features_v2 if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns: {missing_cols}"}), 400

        # Step 5: Extract the data for prediction 
        input_data_for_prediction_v2 = input_df[expected_features_v2].iloc[0].to_dict()  # Extract data as dictionary
        print(f"Data extracted for prediction_v2: {input_data_for_prediction_v2}")  # Debugging

        # Step 6: Use the predictor to get the prediction
        prediction_v2 = predictor_v2.predict(input_data_for_prediction_v2)

        return jsonify({"Predicted Donation Bags with model_v2": prediction_v2})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9000, debug=True)