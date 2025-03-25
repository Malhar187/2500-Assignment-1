# Edmonton City Food Drive

## Project Overview
The **Edmonton City Food Drive** project focuses on implementing an automated system that uses geographic layout and donation totals to assign and optimize pick-up routes. This project aims to improve Edmonton's current food donation management system by addressing the challenges related to drop-off site arrangements, pick-up procedures, and route planning.

## Problem Statement
The existing food donation management system in Edmonton faces several issues:
- **Inefficient drop-off site arrangement**: The drop-off points are not optimally distributed.
- **Suboptimal pick-up procedures**: The current pick-up procedures lack optimization, leading to delays and inefficiencies.
- **Ineffective route planning**: The current routing process does not take into account donation totals, causing inefficiency in transportation.

The goal of this project is to build a system that automates and optimizes the pick-up and routing processes using geographic and donation data.

## Features
- **Route Optimization**: Optimizes pick-up routes based on donation totals and geographic data.
- **Automated Pick-Up Scheduling**: Automatically schedules and assigns routes for food donation pick-ups.
- **Geographic Mapping**: Uses geographic data to ensure efficient routing.

## Technologies Used
- **Python 3.9**: Programming language used for the backend and ML model development.
- **Flask**: Web framework used for building APIs.
- **Machine Learning**:
  - Random Forest Regressor
  - XGBoost
  - ARIMA for time series forecasting
- **Docker**: For containerization of the application.
- **MLflow**: For model tracking, management, and versioning.
- **Docker Compose**: For managing multi-container Docker applications.

## Project Files

- **train.py**: Script for training the machine learning model on historical food donation data.
- **predict.py**: Script to generate predictions using the trained models.
- **predict_api.py**: Flask API that serves the model for real-time predictions.
- **evaluate.py**: Script for evaluating model performance on test data.
- **Dockerfile.mlapp**: Dockerfile for the prediction application container.
- **Dockerfile.mlflow**: Dockerfile for the MLflow container to track and manage model versions.
- **docker-compose.yaml**: Docker Compose configuration for orchestrating the containers (Flask app, MLflow server).
  
## Prerequisites
- Docker
- Python 
- Docker Compose
- Git (for version control)