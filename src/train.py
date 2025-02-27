import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class DonationModelTrainer:
    def __init__(self, data_2023_path: str, data_2024_path: str):
        """Initializes the DonationModelTrainer class by loading datasets."""
        self.data_2023 = pd.read_csv(data_2023_path)
        self.data_2024 = pd.read_csv(data_2024_path)
        self.column_mapping = {
            'TotalRoutes': 'Routes Completed',
            'Time Spent Collecting Donations': 'Time to Complete (min)',
            '# of Adult Volunteers who participated in this route': '# of Adult Volunteers',
            '# of Doors in Route': 'Doors in Route',
            '# of Youth Volunteers who participated in this route': '# of Youth Volunteers',
            'total number of volunteers ': 'Total Volunteers',
            'TotalDonationVolume': 'TotalDonationVolume',
            'Ward/Branch': 'Ward/Branch'
        }
        self.features = ['Routes Completed', 'Time to Complete (min)', '# of Adult Volunteers', 
                         '# of Youth Volunteers', 'Total Volunteers', 'Doors in Route']
        self.target = 'TotalDonationVolume'
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepares and renames data columns for consistency."""
        self.data_2023.rename(columns=self.column_mapping, inplace=True)
        self.data_2024.rename(columns=self.column_mapping, inplace=True)

        self.data_2023 = self.data_2023.loc[:, ~self.data_2023.columns.duplicated()]
        self.data_2024 = self.data_2024.loc[:, ~self.data_2024.columns.duplicated()]

    def train_models(self):
        """Trains Linear Regression and Random Forest models, logging results in MLflow."""
        X = self.data_2023[self.features]
        y = self.data_2023[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Keep input_example as a pandas DataFrame (to retain feature names)
        input_example = X_train.iloc[[0]]
        # Start MLflow Experiment
        mlflow.set_experiment("Donation_Prediction_Experiment")

        with mlflow.start_run():
            # Train Linear Regression Model
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)

            # Train Random Forest Model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            # Compute Evaluation Metrics
            metrics = {
                "lr_mse": mean_squared_error(y_test, y_pred_lr),
                "lr_r2": r2_score(y_test, y_pred_lr),
                "lr_mae": mean_absolute_error(y_test, y_pred_lr),
                "rf_mse": mean_squared_error(y_test, y_pred_rf),
                "rf_r2": r2_score(y_test, y_pred_rf),
                "rf_mae": mean_absolute_error(y_test, y_pred_rf)
            }

            # Log Parameters
            mlflow.log_param("n_estimators_rf", 100)

            # Log Metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log Models with feature names
            mlflow.sklearn.log_model(lr_model, "Linear_Regression_Model", input_example=input_example)
            mlflow.sklearn.log_model(rf_model, "Random_Forest_Model", input_example=input_example)

            print(f"Metrics logged in MLflow: {metrics}")

        return y_pred_lr, y_pred_rf

# Usage
data_2023_path = 'Data/processed/data_2023_donation_processed.csv'
data_2024_path = 'Data/processed/data_2024_donation_processed.csv'

donation_trainer = DonationModelTrainer(data_2023_path, data_2024_path)
y_pred_lr, y_pred_rf = donation_trainer.train_models()