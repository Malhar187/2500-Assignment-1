-- **Food Drive API Documentation** --

# OVERVIEW:
This Food Drive API is designed to predict donation bag collection based on various fctors such as completed routes, volunteer data, and time spent collecting donations. The API supports multiple models to improve accuracy and handle different prediction scenarios. 

# BASE URL: 
http://0.0.0.0:5000

# ENDPOINTS: 
1. Home Endpoint -
GET /food_drive_home
This returns general overview and information about the API. 

*Response:*
{
    "name": "Food Drive API",
    "description": "This API takes in different inputs and gives donation predictions.",
    "version": "v1.0",
    "endpoints": {
        "/food_drive_home": {
            "description": "Home Page - Provides API documentation.",
            "method": "GET"
        },
        "/health_status": {
            "description": "Check API's health status.",
            "method": "GET",
            "response": {
                "status": "API is running",
                "message": "Healthy"
            }
        },
        "/predict/v1": {
            "description": "Predicts donation bags using model_v1.",
            "method": "POST",
            "expected_input": [
                "Routes Completed", "Time to Complete (min)", "Doors in Route",
                "AvgDoorsPerRoute", "Total Volunteers", "# of Adult Volunteers",
                "# of Youth Volunteers", "VolunteersPerDoor", "TimePerVolunteer"
            ]
        },
        "/predict/v2": {
            "description": "Predicts donation bags using model_v2.",
            "method": "POST",
            "expected_input": [
                "AvgDoorsPerRoute", "Time to Complete (min)", "Doors in Route",
                "Routes Completed", "# of Adult Volunteers"
            ]
        }
    }
}

2. Health Check Endpoint (/health_status) - 
Method: GET
This endpoint checks if the API is running and healthy. 

*Response:*
{
    "status": "API is running",
    "message": "Healthy"
}

3. Predictions using Model v1 (/predict/v1) - 
Method: POST
This endpoint uses model_v1 to predict the number of donation bags collected. This requires input data in JSON format.

*Expected input in JSON format:*
{
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

*Response:*
{
    "Predicted Donation Bags with model_v1": 35
}

4. Predictions using Model v2 (/predict/v2) - 
Method: POST
This endpoint uses model_v2 to predict the number of donation bags collected. This requires input data in JSON format, and the input data is different than model_v1.

*Expected input in JSON format:*
{
    "AvgDoorsPerRoute": 10,
    "Time to Complete (min)": 120,
    "Doors in Route": 25,
    "Routes Completed": 5,
    "# of Adult Volunteers": 3
}

*Response:*
{
    "Predicted Donation Bags with model_v2": 28
}

# ERROR HANDLING: 
The API returns appropriate error messages in case of incorrect input formats or missing data.

*Example Error Response (Missing Required Fields):*
{
    "error": "Missing required columns: ['Total Volunteers']"
}

*Example Error Response (Invalid Format):*
{
    "error": "Invalid input format. Send a list of dictionaries or a single dictionary."
}

# USAGE EXAMPLES:
*cURL Example for Model v1 Prediction:*

curl -X POST "http://127.0.0.1:5001/predict/v1" \
     -H "Content-Type: application/json" \
     -d '{
            "Routes Completed": 5,
            "Time to Complete (min)": 120,
            "Doors in Route": 25,
            "AvgDoorsPerRoute": 10,
            "Total Volunteers": 5,
            "# of Adult Volunteers": 3,
            "# of Youth Volunteers": 2,
            "VolunteersPerDoor": 3,
            "TimePerVolunteer": 5
        }'

*cURL Example for Model v2 Prediction:*

curl -X POST "http://127.0.0.1:5001/predict/v2" \
     -H "Content-Type: application/json" \
     -d '{
            "AvgDoorsPerRoute": 10,
            "Time to Complete (min)": 120,
            "Doors in Route": 25,
            "Routes Completed": 5,
            "# of Adult Volunteers": 3
        }'

# CONCLUSION:
This API enables efficient donation predictions for food drives using trained machine learning models. Make sure to provide the correct input format when making POST requests to /predict/v1 and /predict/v2. If you encounter any issues, check the health status endpoint, or check the expected shape of the dataframe before passing it to the model. 


