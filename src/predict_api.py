from flask import Flask, request, jsonify
import joblib
import pandas as pd
from predict import DonationModelPredictor
import logging
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import time
import psutil
import threading
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Custom Prometheus metrics
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version', 'status'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

# Initialize model paths
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
    start_time = time.time()
    model_version = "v1"
    
    try:
        # Step 1: Get input JSON from the request
        input_data = request.get_json()
        logger.debug(f"Received input data: {input_data}")  # Logging received data

        # Step 2: Ensure input_data is a list of dictionaries
        if isinstance(input_data, dict):
            input_data = [input_data]  # If it's a single dictionary, convert to list

        logger.debug(f"Data after ensuring list format: {input_data}")  # Logging the list format
        
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
            logger.error(f"Missing required columns: {missing_cols}")  # Logging the error
            return jsonify({"error": f"Missing required columns: {missing_cols}"}), 400

        # Step 5: Extract the data for prediction (assuming input_df has the correct columns)
        input_data_for_prediction = input_df[expected_features].iloc[0].to_dict()  # Extract data as dictionary
        logger.debug(f"Data extracted for prediction: {input_data_for_prediction}")  # Logging extracted data

        # Step 6: Use the predictor to get the prediction
        prediction = predictor_v1.predict(input_data_for_prediction)

        logger.info(f"Prediction result: {prediction}")  # Logging the prediction result
        
        # Record prediction time
        prediction_requests.labels(model_version=model_version, status="success").inc()
        prediction_time.labels(model_version=model_version).observe(time.time() - start_time)
        
        return jsonify({"Predicted Donation Bags with model_v1": prediction})
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")  # Logging the error
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict/v2", methods=["POST"])
def predict_v2():
    start_time = time.time()
    model_version = "v2"
    
    try:
        # Step 1: Get input JSON from the request
        input_data = request.get_json()
        logger.debug(f"Received input data: {input_data}")  # Logging received data

        # Step 2: Ensure input_data is a list of dictionaries
        if isinstance(input_data, dict):
            input_data = [input_data]  # If it's a single dictionary, convert to list

        logger.debug(f"Data after ensuring list format: {input_data}")  # Logging the list format
        
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
            logger.error(f"Missing required columns: {missing_cols}")  # Logging the error
            return jsonify({"error": f"Missing required columns: {missing_cols}"}), 400

        # Step 5: Extract the data for prediction 
        input_data_for_prediction_v2 = input_df[expected_features_v2].iloc[0].to_dict()  # Extract data as dictionary
        logger.debug(f"Data extracted for prediction_v2: {input_data_for_prediction_v2}")  # Logging extracted data

        # Step 6: Use the predictor to get the prediction
        prediction_v2 = predictor_v2.predict(input_data_for_prediction_v2)

        logger.info(f"Prediction result for model_v2: {prediction_v2}")  # Logging the prediction result
        
        # Record prediction time
        prediction_requests.labels(model_version=model_version, status="success").inc()
        prediction_time.labels(model_version=model_version).observe(time.time() - start_time)

        return jsonify({"Predicted Donation Bags with model_v2": prediction_v2})
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")  # Logging the error
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": str(e)}), 500

# Add a function to monitor resource usage in the background
def monitor_resources():
    """Update system resource metrics every 15 seconds"""
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)  # in bytes
        cpu_usage.set(process.cpu_percent())
        time.sleep(15)

# Start resource monitoring in a background thread when app starts
if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)