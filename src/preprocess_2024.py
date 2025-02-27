import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

class FoodDriveData2024:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        pd.set_option('display.max_rows', 100)
        print(self.df.info())
    
    def check_duplicates_and_nulls(self):
        print("Duplicate rows:", self.df.duplicated().sum())
        print("Missing values per column:\n", self.df.isnull().sum())
    
    def generate_route_names(self):
        route_counters = {}
        new_route_names = []

        for _, row in self.df.iterrows():
            stake = row['Stake']
            if stake not in route_counters:
                route_counters[stake] = 1
            else:
                route_counters[stake] += 1
            new_route_names.append(f"{stake} {route_counters[stake]}")

        self.df['New Route Number/Name'] = new_route_names

# Usage
if __name__ == "__main__":
    food_drive_2024 = FoodDriveData2024("data_2024_dontation.csv")
    food_drive_2024.load_data()
    food_drive_2024.check_duplicates_and_nulls()
    food_drive_2024.generate_route_names()
