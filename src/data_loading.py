#data_loading.py
"""
Simplified data loading module for wildfire prediction project.
Loads pre-filtered California wildfire data and NOAA weather data.
"""
import os
import pandas as pd
import numpy as np
import argparse

def load_wildfire_data(year, file_path='../data/raw/ca_daily_fire_2000_2021-v2.csv'):
    """Load pre-filtered California wildfire data for a specific year."""
    try:
        # Load the consolidated California wildfires file
        df = pd.read_csv(file_path)
        
        # Filter to the requested year
        year_df = df[df['year'] == year]
        
        print(f"Loaded {len(year_df)} wildfire events in California for {year}")
        
        # Create datetime column (combining date with midnight time since no time in data)
        year_df['datetime'] = pd.to_datetime(year_df['acq_date'])
        
        return year_df
    except Exception as e:
        print(f"Error loading wildfire data: {e}")
        return None

def load_weather_data(year):
    """Load NOAA GSOD weather data for a specific year."""
    file_path = f'../csv_gsod/{year}/'
    
    # Find the CSV file in the directory
    files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
    
    if not files:
        print(f"No CSV files found in {file_path}")
        return None
    
    print(f"Loading weather data from {files[0]}")
    # Load the first CSV file found
    df = pd.read_csv(os.path.join(file_path, files[0]), low_memory=False)
    
    # Convert date to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Extract unique weather stations
    stations = df[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    print(f"Found {len(stations)} unique weather stations")
    
    # Convert numeric columns to float, handling errors
    numeric_cols = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace missing values (999.9) with NaN
    df.replace(999.9, np.nan, inplace=True)
    
    print(f"Loaded {len(df)} weather records for {year}")
    return df, stations

def save_processed_data(df, name, year):
    """Save processed data to CSV."""
    output_dir = f'../data/processed/{year}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{name}_{year}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {name} data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load wildfire and weather data for analysis')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2019)')
    parser.add_argument('--wildfire_file', type=str, default='../data/raw/ca_daily_fire_2000_2021-v2.csv', 
                        help='Path to consolidated California wildfires file')
    args = parser.parse_args()
    
    year = args.year
    print(f"Processing data for year {year}")
    
    # Load data
    wildfires = load_wildfire_data(year, args.wildfire_file)
    weather, stations = load_weather_data(year)
    
    # Save processed data
    if wildfires is not None:
        save_processed_data(wildfires, 'wildfires', year)
    if weather is not None:
        save_processed_data(weather, 'weather', year)
        save_processed_data(stations, 'stations', year)
    
    print("Data loading complete!")