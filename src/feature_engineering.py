#feature_engineering.py

"""
Simplified feature engineering for wildfire prediction project.
"""
import os
import pandas as pd
import numpy as np
import argparse
from scipy.spatial import cKDTree
from datetime import timedelta
def load_data(year):
    """Load processed data for feature engineering."""
    data_dir = f'../data/processed/{year}'
    
    try:
        # Change to load wildfire events instead of individual detections
        wildfire_events = pd.read_csv(os.path.join(data_dir, f'wildfire_events_{year}.csv'))
        wildfire_events['start_date'] = pd.to_datetime(wildfire_events['start_date'])
        wildfire_events['end_date'] = pd.to_datetime(wildfire_events['end_date'])
        
        weather = pd.read_csv(os.path.join(data_dir, f'weather_{year}.csv'))
        weather['DATE'] = pd.to_datetime(weather['DATE'])
        
        stations = pd.read_csv(os.path.join(data_dir, f'stations_{year}.csv'))
        
        return wildfire_events, weather, stations
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def create_weather_features(wildfire_events_df, weather_df, stations_df, days_lookback=30):
    """
    Create weather features for each wildfire event by looking back at weather data
    from the nearest station for the specified number of days before the start date.
    """
    if wildfire_events_df is None or weather_df is None or stations_df is None:
        print("Missing data for feature engineering")
        return None
    
    print("Creating weather features for wildfire events...")
    
    # Create KDTree for efficient nearest station lookup
    tree = cKDTree(stations_df[['LATITUDE', 'LONGITUDE']].values)
    
    # Create empty dataframe to store results
    result_rows = []
    
    # Process each wildfire event
    for idx, fire in wildfire_events_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing wildfire event {idx}/{len(wildfire_events_df)}")
        
        # Find nearest weather station
        distance, index = tree.query([fire['latitude'], fire['longitude']])
        station_id = stations_df.iloc[index]['STATION']
        station_name = stations_df.iloc[index]['NAME']
        
        # Get fire start date
        fire_date = fire['start_date']
        
        # Get weather data for the preceding days
        station_weather = weather_df[weather_df['STATION'] == station_id]
        past_weather = station_weather[
            (station_weather['DATE'] < fire_date) & 
            (station_weather['DATE'] >= (fire_date - timedelta(days=days_lookback)))
        ]
        
        # Skip if no weather data available
        if len(past_weather) < 5:  
            continue
        
        # Calculate features from weather history
        features = {
            'event_id': fire['event_id'],
            'fire_start_date': fire['start_date'],
            'fire_end_date': fire['end_date'],
            'fire_duration_days': fire['duration_days'],
            'fire_latitude': fire['latitude'],
            'fire_longitude': fire['longitude'],
            'fire_max_frp': fire['max_frp'],
            'fire_avg_frp': fire['avg_frp'],
            'fire_total_frp': fire['total_frp'],
            'confidence': fire['max_confidence'],
            'station_id': station_id,
            'station_name': station_name,
            'station_distance_km': distance * 111,  # Approx conversion from degrees to km
            
            # Temperature features
            'avg_temp_30d': past_weather['TEMP'].mean(),
            'max_temp_30d': past_weather['MAX'].max(),
            'avg_max_temp_30d': past_weather['MAX'].mean(),
            
            # Precipitation features
            'total_precip_30d': past_weather['PRCP'].sum(),
            'days_without_rain': sum(past_weather['PRCP'] < 0.01),
            
            # Wind features
            'avg_wind_speed': past_weather['WDSP'].mean(),
            'max_wind_speed': past_weather['MXSPD'].max(),
            
            # Humidity related
            'avg_dewpoint': past_weather['DEWP'].mean(),
            
            # Derived features
            'temp_precip_ratio': past_weather['TEMP'].mean() / (past_weather['PRCP'].sum() + 0.1),
        }
        
        result_rows.append(features)
    
    # Create dataframe from results
    if result_rows:
        result_df = pd.DataFrame(result_rows)
        print(f"Created features for {len(result_df)} wildfire events")
        return result_df
    else:
        print("No features created, check data compatibility")
        return None
    
def save_features(df, year):
    """Save feature data to CSV."""
    output_dir = f'../data/features/{year}'
    os.makedirs(output_dir, exist_ok=True)
    # Updated filename to reflect event-based features
    output_path = os.path.join(output_dir, f'wildfire_event_features_{year}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved feature data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create features for wildfire prediction')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2019)')
    parser.add_argument('--lookback', type=int, default=30, help='Days of weather history to use')
    args = parser.parse_args()
    
    year = args.year
    lookback_days = args.lookback
    
    # Load data
    wildfires, weather, stations = load_data(year)
    
    # Create features
    if wildfires is not None and weather is not None and stations is not None:
        features = create_weather_features(wildfires, weather, stations, days_lookback=lookback_days)
        
        # Save features
        if features is not None and not features.empty:
            save_features(features, year)
    
    print("Feature engineering complete!")