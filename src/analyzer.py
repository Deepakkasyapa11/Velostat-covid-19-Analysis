import pandas as pd
import numpy as np
import os

class CovidAnalyzer:
    """High-performance processing engine for epidemiological time-series."""
    
    def __init__(self, file_name="time-series-19-covid-combined.csv"):
        # Dynamic path discovery
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.raw_path = os.path.join(base_path, "data", "raw", file_name)
        
        self.df = pd.read_csv(self.raw_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def get_country_metrics(self, country: str):
        """Calculates growth velocity using NumPy convolution."""
        data = self.df[self.df['Country/Region'] == country].copy().sort_values('Date')
        confirmed = data['Confirmed'].values
        
        # Calculate Daily Velocity (New Cases)
        daily_new = np.diff(confirmed, prepend=0)
        daily_new = np.maximum(daily_new, 0) # Clean reporting anomalies
        
        # 7-Day Smoothing (Signal processing approach)
        window = 7
        moving_avg = np.convolve(daily_new, np.ones(window)/window, mode='same')
        
        data['Daily_New'] = daily_new
        data['7Day_MA'] = moving_avg
        return data

if __name__ == "__main__":
    analyzer = CovidAnalyzer()
    india_stats = analyzer.get_country_metrics("India")
    print(f"Ingestion complete. Latest smoothed cases for India: {india_stats['7Day_MA'].iloc[-1]:.2f}")