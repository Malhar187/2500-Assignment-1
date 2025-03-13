import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

class DonationModel:
    def __init__(self, model_path: str, data_path: str):
        """Initializes the DonationModel class by loading the trained model and dataset."""
        self.model = joblib.load(model_path)
        self.data = pd.read_csv(data_path)
        self.prepare_data()
    
    def prepare_data(self):
        """Renames columns and creates new feature columns."""
        column_mapping = {
            'TotalRoutes': 'Routes Completed',
            'Time Spent Collecting Donations': 'Time to Complete (min)',
            '# of Adult Volunteers who participated in this route': '# of Adult Volunteers',
            '# of Doors in Route': 'Doors in Route',
            '# of Youth Volunteers who participated in this route': '# of Youth Volunteers',
            'total number of volunteers ': 'Total Volunteers',
            'TotalDonationVolume': 'TotalDonationVolume',
            'Ward/Branch': 'Ward/Branch',
            'AvgDoorsPerRoute': 'AvgDoorsPerRoute',
            '# of Donation Bags Collected': 'Donation Bags Collected'
        }
        self.data = self.data.rename(columns=column_mapping)
        self.data['VolunteersPerDoor'] = self.data['Total Volunteers'] / self.data['Doors in Route']
        self.data['TimePerVolunteer'] = self.data['Time to Complete (min)'] / self.data['Total Volunteers']
        self.data['DonationBagsPerVolunteer'] = self.data['Donation Bags Collected'] / self.data['Total Volunteers']
    
    def predict(self):
        """Predicts donation bags collected using the trained model."""
        features = ['Routes Completed', 'Time to Complete (min)', '# of Adult Volunteers', 'Doors in Route',
                    '# of Youth Volunteers', 'Total Volunteers', 'TotalDonationVolume', 'Doors in Route',
                    'AvgDoorsPerRoute', 'Total Volunteers', 'VolunteersPerDoor', 'TimePerVolunteer']
        target = 'Donation Bags Collected'
        
        X = self.data[features]
        y = self.data[target]
        y_pred = self.model.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        print(f'Mean Squared Error: {mse}')
        print(f'R2 Score: {r2}')
        print(f'Mean Absolute Error: {mae}')
        
        return y_pred

# Usage
model_path = 'trained_donation_model.pkl'
data_path = 'data_2024_donation_processed.csv'
donation_model = DonationModel(model_path, data_path)
donation_model.predict()
