import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)

class CovidAnalyticsEngine:
    def __init__(self, raw_data_path: str):
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Data file not found at {raw_data_path}")
        self.df = pd.read_csv(raw_data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        logging.info(f"Ingested {len(self.df)} records.")

    def get_velocity_metrics(self, country: str):
        subset = self.df[self.df['Country/Region'] == country].copy().sort_values('Date')
        if subset.empty:
            logging.error(f"No data found for country: {country}")
            return None
            
        confirmed = subset['Confirmed'].values
        daily_new = np.diff(confirmed, prepend=0)
        daily_new = np.maximum(daily_new, 0) 
        
        window = 7
        moving_avg = np.convolve(daily_new, np.ones(window)/window, mode='same')
        
        subset['Daily_New'] = daily_new
        subset['7Day_MA'] = moving_avg
        return subset

if __name__ == "__main__":
    # Adjust path to match your structure
    DATA_PATH = "data/raw/time-series-19-covid-combined.csv"
    engine = CovidAnalyticsEngine(DATA_PATH)
    results = engine.get_velocity_metrics("India")
    if results is not None:
        logging.info(f"Latest Growth Velocity in India: {results['Daily_New'].iloc[-1]}")