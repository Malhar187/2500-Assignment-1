# Food Drive Machine Learning Project

# Overview

To learn data processing and cleaning and enhancing data collection skills. 

Create a machine learning model to determine the ideal drop-off sites in accordance with donation density and geographic distribution.

Put in place an automated system that uses the geographic layout and donation totals to assign and optimize pick-up routes.

Stakeholder Food Drive Representatives, Ward Food Drive Representatives, and Regional Coordinators should all communicate and coordinate more efficiently.

Improve data gathering and analysis to learn more about resource usage, donation trends, and areas for development.

Project Structure:

1. Preprocessing (preprocess.py) – Cleans and prepares the raw data for analysis (individually for years 2023 and 2024).
  
2. Training (train.py) – Trains machine learning models using the processed data.
  
3. Evaluation (evaluate.py) – Assesses the performance of the trained models on test data.
  
4. Feature Importance (Feature Importance.py) – Identifies the most important features within the trained models.
  
5. Prediction (predict.py) – Utilizes the trained model to make predictions on new data.



1.  (preprocess.py)

This script loads the raw dataset, performs data cleaning, handles missing values, merges external data like property assessments, and applies necessary transformations. The processed data is then saved to the data/processed/ directory for further use.


2. (train.py)

This script loads the processed data, trains multiple machine learning models (including Linear Regression, Decision Tree, Random Forest, and XGBoost), and conducts hyperparameter tuning to optimize model performance. Additionally, it evaluates model accuracy and saves the trained models in the models/ directory for future use and deployment.


3.(evaluate.py)

This script loads the trained models and assesses their performance on test data using evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared Score.


4 (Feature Importance.py)

This script identifies the key features that have the greatest impact on the predictions made by the trained model. It provides insights into which factors are most influential in forecasting donation volumes.


5.  (predict.py)

This script utilizes the trained model to make predictions on new, unseen data. It processes the input data, applies the model, and outputs the predicted results for future donation volumes or outcomes.