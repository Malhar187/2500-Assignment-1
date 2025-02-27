import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load the datasets
data_2023 = pd.read_csv('data_2023_donation_processed.csv')
data_2024 = pd.read_csv('data_2024_donation_processed.csv')

# Define features and target
features = ['Routes Completed', 'Time to Complete (min)', '# of Adult Volunteers', '# of Youth Volunteers', 'Total Volunteers']
target = 'TotalDonationVolume'

X = data_2023[features]
y = data_2023[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print(feature_importance_df)
