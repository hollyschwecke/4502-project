#!/usr/bin/env python3
"""
Preprocessing module for wildfire prediction project.
Preprocesses MODIS satellite wildfire data and NOAA GSOD weather data.
"""
import os
import pandas as pd
import numpy as np
import argparse

def preprocess_wildfire_data(wildfire_df):
    """Preprocess wildfire data."""
    if wildfire_df is None or wildfire_df.empty:
        print("No wildfire data to preprocess")
        return None
    
    print("Preprocessing wildfire data...")
    # Extract month and day for seasonality analysis
    wildfire_df['month'] = wildfire_df['datetime'].dt.month
    wildfire_df['day'] = wildfire_df['datetime'].dt.day
    
    # Create a confidence level category (low, medium, high)
    bins = [0, 30, 70, 100]
    labels = ['low', 'medium', 'high']
    wildfire_df['confidence_level'] = pd.cut(wildfire_df['confidence'], bins=bins, labels=labels)
    
    # Fire radiative power (FRP) is an indicator of fire intensity
    # Create categories for analysis
    wildfire_df['frp_category'] = pd.qcut(wildfire_df['frp'], q=4, labels=['low', 'medium-low', 'medium-high', 'high'])
    
    print("Wildfire preprocessing complete")
    return wildfire_df

def preprocess_weather_data(weather_df):
    """Preprocess weather data."""
    if weather_df is None or weather_df.empty:
        print("No weather data to preprocess")
        return None
    
    print("Preprocessing weather data...")
    # Extract date components
    weather_df['month'] = weather_df['DATE'].dt.month
    weather_df['day'] = weather_df['DATE'].dt.day
    
    # Calculate drought indicators
    # Simple drought index: difference between max temp and precipitation
    weather_df['drought_index'] = weather_df['MAX'] - (weather_df['PRCP'] * 100)
    
    # Wind danger index: combination of wind speed and low humidity
    # Lower dewpoint means lower humidity
    weather_df['wind_danger'] = weather_df['MXSPD'] * (weather_df['TEMP'] - weather_df['DEWP'])
    
    # Fill missing values with median for each station
    print("Filling missing values...")
    for station in weather_df['STATION'].unique():
        station_data = weather_df[weather_df['STATION'] == station]
        for col in ['TEMP', 'DEWP', 'WDSP', 'PRCP', 'MAX', 'MIN']:
            median_val = station_data[col].median()
            weather_df.loc[(weather_df['STATION'] == station) & (weather_df[col].isna()), col] = median_val
    
    print("Weather preprocessing complete")
    return weather_df

def load_data(year):
    """Load raw data for preprocessing."""
    data_dir = f'../data/processed/{year}'
    
    try:
        wildfires = pd.read_csv(os.path.join(data_dir, f'wildfires_raw_{year}.csv'))
        wildfires['datetime'] = pd.to_datetime(wildfires['datetime'])
        
        weather = pd.read_csv(os.path.join(data_dir, f'weather_raw_{year}.csv'))
        weather['DATE'] = pd.to_datetime(weather['DATE'])
        
        return wildfires, weather
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def save_processed_data(df, name, year):
    """Save processed data to CSV."""
    output_dir = f'../data/processed/{year}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{name}_{year}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {name} data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess wildfire and weather data')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2019)')
    args = parser.parse_args()
    
    year = args.year
    print(f"Preprocessing data for year {year}")
    
    # Load raw data
    wildfires, weather = load_data(year)
    
    # Preprocess data
    processed_wildfires = preprocess_wildfire_data(wildfires)
    processed_weather = preprocess_weather_data(weather)
    
    # Save processed data
    if processed_wildfires is not None:
        save_processed_data(processed_wildfires, 'wildfires_processed', year)
    if processed_weather is not None:
        save_processed_data(processed_weather, 'weather_processed', year)
    
    print("Preprocessing complete!")