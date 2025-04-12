import pandas as pd
import mlflow
import mlflow.sklearn
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

from utils.monitoring import RegressionMonitor, TreeModelMonitor

# Set up logging
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{log_dir}/train.log")
    ]
)

logger = logging.getLogger("ml_app.train")

class DonationModelTrainer:
    def __init__(self, data_2023_path: str, data_2024_path: str):
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
        logger.info("Preparing data by renaming columns and removing duplicates.")
        self.data_2023.rename(columns=self.column_mapping, inplace=True)
        self.data_2024.rename(columns=self.column_mapping, inplace=True)
        self.data_2023 = self.data_2023.loc[:, ~self.data_2023.columns.duplicated()]
        self.data_2024 = self.data_2024.loc[:, ~self.data_2024.columns.duplicated()]

    def train_models(self):
        logger.info("Starting model training process.")
        X = self.data_2023[self.features]
        y = self.data_2023[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        input_example = X_train.iloc[[0]]

        # Initialize monitors
        lr_monitor = RegressionMonitor(port=8002)
        rf_monitor = TreeModelMonitor(port=8003)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("Donation_Prediction_Experiment")
        with mlflow.start_run():
            logger.info("Training Linear Regression Model.")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)

            # Evaluate and monitor LR
            lr_mse = mean_squared_error(y_test, y_pred_lr)
            lr_r2 = r2_score(y_test, y_pred_lr)
            lr_mae = mean_absolute_error(y_test, y_pred_lr)
            lr_feature_importance = dict(zip(self.features, abs(lr_model.coef_)))

            lr_monitor.record_metrics(
                mse=lr_mse,
                rmse=lr_mse**0.5,
                mae=lr_mae,
                r_squared=lr_r2,
                feature_importance=lr_feature_importance
            )

            logger.info("Training Random Forest Model.")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            # Evaluate and monitor RF
            rf_mse = mean_squared_error(y_test, y_pred_rf)
            rf_r2 = r2_score(y_test, y_pred_rf)
            rf_mae = mean_absolute_error(y_test, y_pred_rf)
            rf_feature_importance = dict(zip(self.features, rf_model.feature_importances_))

            rf_monitor.record_metrics(
                mse=rf_mse,
                rmse=rf_mse**0.5,
                mae=rf_mae,
                r_squared=rf_r2,
                feature_importance=rf_feature_importance
            )
            rf_monitor.record_tree_metrics(
                depth=max(estimator.tree_.max_depth for estimator in rf_model.estimators_),
                leaves=sum(estimator.tree_.n_leaves for estimator in rf_model.estimators_),
                trees=len(rf_model.estimators_)
            )

            metrics = {
                "lr_mse": lr_mse,
                "lr_r2": lr_r2,
                "lr_mae": lr_mae,
                "rf_mse": rf_mse,
                "rf_r2": rf_r2,
                "rf_mae": rf_mae
            }

            logger.info(f"Metrics computed: {metrics}")
            mlflow.log_param("n_estimators_rf", 100)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(lr_model, "Linear_Regression_Model", input_example=input_example)
            mlflow.sklearn.log_model(rf_model, "Random_Forest_Model", input_example=input_example)

            logger.info("Models and metrics logged successfully in MLflow.")

        return y_pred_lr, y_pred_rf

# Usage
data_2023_path = 'Data/processed/data_2023_donation_processed.csv'
data_2024_path = 'Data/processed/data_2024_donation_processed.csv'

logger.info("Initializing DonationModelTrainer.")
donation_trainer = DonationModelTrainer(data_2023_path, data_2024_path)
y_pred_lr, y_pred_rf = donation_trainer.train_models()