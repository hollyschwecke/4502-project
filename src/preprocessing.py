"""
Consolidated data loading and preprocessing module for wildfire prediction project.
Loads and preprocesses California wildfire data and NOAA GSOD weather data.
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

def load_weather_data(year, fire_locations=None, radius_km=50):
    """
    Load NOAA GSOD weather data for stations near fire events.
    
    Parameters:
    -----------
    year : int
        Year to load weather data for
    fire_locations : DataFrame, optional
        DataFrame containing latitude and longitude of fire events. 
        If provided, will filter stations to only those near fires.
    radius_km : float, optional
        Maximum distance in kilometers to consider a station "near" a fire event
    
    Returns:
    --------
    weather_df : DataFrame
        Weather data from stations near fire events
    stations_df : DataFrame
        Information about the stations included in the weather data
    """
    file_path = f'../csv_gsod/{year}/'
    
    try:
        # Find all CSV files in the directory
        files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
        
        if not files:
            print(f"No CSV files found in {file_path}")
            return None, None
        
        print(f"Found {len(files)} weather data files for {year}")
        
        # First, just load the station metadata from the first file
        # to get station locations without loading all weather data
        print("Loading station metadata...")
        first_file = files[0]
        sample_df = pd.read_csv(os.path.join(file_path, first_file), low_memory=False, nrows=1000)
        all_stations = sample_df[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
        
        # For remaining files, get unique stations
        for file in files[1:]:
            try:
                sample = pd.read_csv(os.path.join(file_path, file), low_memory=False, nrows=1000)
                stations = sample[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
                all_stations = pd.concat([all_stations, stations], ignore_index=True).drop_duplicates()
            except Exception as e:
                print(f"Warning: Error loading station data from {file}: {e}")
        
        print(f"Found {len(all_stations)} unique weather stations")
        
        # Filter stations if fire locations are provided
        relevant_stations = all_stations.copy()
        if fire_locations is not None and len(fire_locations) > 0:
            print(f"Filtering stations based on proximity to {len(fire_locations)} fire events...")
            
            # Convert radius to approximate degrees
            # Rough approximation: 1 degree latitude ≈ 111 km
            radius_deg = radius_km / 111.0
            
            # Create a KDTree for efficient spatial querying
            from scipy.spatial import cKDTree
            
            # Create a KDTree with all station locations
            station_coords = all_stations[['LATITUDE', 'LONGITUDE']].values
            tree = cKDTree(station_coords)
            
            # Create a set to track relevant station IDs
            relevant_station_ids = set()
            
            # For each fire, find nearby stations
            for idx, fire in fire_locations.iterrows():
                if idx % 100 == 0 and idx > 0:
                    print(f"  Processed {idx}/{len(fire_locations)} fire events...")
                
                # Query for stations within radius
                indices = tree.query_ball_point(
                    [fire['latitude'], fire['longitude']], 
                    radius_deg
                )
                
                # Add these station IDs to our set
                for i in indices:
                    relevant_station_ids.add(all_stations.iloc[i]['STATION'])
            
            # Filter to just the relevant stations
            relevant_stations = all_stations[all_stations['STATION'].isin(relevant_station_ids)]
            print(f"Filtered to {len(relevant_stations)} stations within {radius_km}km of fire events")
        
        # Now load the actual weather data, but only for relevant stations
        print("Loading weather data for relevant stations...")
        weather_dfs = []
        station_ids = relevant_stations['STATION'].unique()
        
        for file in files:
            try:
                # Read in chunks to filter by station ID efficiently
                chunk_size = 100000  # Adjust based on memory constraints
                chunks = pd.read_csv(os.path.join(file_path, file), low_memory=False, chunksize=chunk_size)
                
                for chunk in chunks:
                    # Filter to relevant stations
                    filtered_chunk = chunk[chunk['STATION'].isin(station_ids)]
                    if not filtered_chunk.empty:
                        weather_dfs.append(filtered_chunk)
            except Exception as e:
                print(f"Warning: Error loading weather data from {file}: {e}")
        
        # Combine all data
        if weather_dfs:
            weather_df = pd.concat(weather_dfs, ignore_index=True)
            print(f"Loaded {len(weather_df)} weather records for {len(station_ids)} stations")
            
            # Convert date to datetime
            weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
            
            # Convert numeric columns to float, handling errors
            numeric_cols = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP']
            for col in numeric_cols:
                if col in weather_df.columns:
                    weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
            
            # Replace missing values (999.9) with NaN
            weather_df.replace(999.9, np.nan, inplace=True)
            
            return weather_df, relevant_stations
        else:
            print("No weather data found for relevant stations")
            return None, relevant_stations
        
    except Exception as e:
        print(f"Error loading weather data: {e}")
        return None, None

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

def cluster_wildfire_events(wildfire_df, distance_threshold_km=10):
    """
    Cluster wildfire data points into fire events based on:
    1. Spatial proximity (within distance_threshold_km)
    2. Temporal proximity (same month and year)
    
    Returns a dataframe with clustered events.
    """
    if wildfire_df is None or wildfire_df.empty:
        print("No wildfire data to cluster")
        return None
    
    print("Clustering wildfire data points into events...")
    
    # Create a copy to avoid modifying the original
    df = wildfire_df.copy()
    
    # Ensure datetime column exists and is datetime type
    if 'datetime' not in df.columns:
        print("Error: datetime column required for clustering")
        return None
    
    # Extract year and month for temporal grouping
    df['year_month'] = df['datetime'].dt.to_period('M')
    
    # Initialize event ID column
    df['event_id'] = -1
    next_event_id = 0
    
    # Group by year and month first
    for year_month, group in df.groupby('year_month'):
        print(f"Processing {year_month} with {len(group)} detections")
        
        # Create list of indices in this group to track which are assigned
        unassigned = group.index.tolist()
        
        # Process until all points in this time period are assigned
        while unassigned:
            # Take first unassigned point as seed for new event
            seed_idx = unassigned[0]
            seed_point = group.loc[seed_idx]
            
            # Create new event
            df.loc[seed_idx, 'event_id'] = next_event_id
            unassigned.remove(seed_idx)
            
            # Find all points within distance threshold of this seed
            points_in_event = [seed_idx]
            
            # Spatial clustering
            for idx in unassigned.copy():
                point = group.loc[idx]
                # Calculate approximate distance in km
                lat_diff = point['latitude'] - seed_point['latitude']
                lon_diff = point['longitude'] - seed_point['longitude']
                # Simple approximation: 1 degree ~= 111km
                distance_km = np.sqrt((lat_diff * 111)**2 + (lon_diff * 111 * np.cos(np.radians(seed_point['latitude'])))**2)
                
                if distance_km <= distance_threshold_km:
                    # Add to current event
                    df.loc[idx, 'event_id'] = next_event_id
                    points_in_event.append(idx)
                    unassigned.remove(idx)
            
            print(f"  Created event {next_event_id} with {len(points_in_event)} detections")
            next_event_id += 1
    
    # Compute event-level aggregated data
    print(f"Created {next_event_id} distinct fire events")
    
    # Aggregate data at the event level
    event_data = []
    
    for event_id, event_group in df[df['event_id'] >= 0].groupby('event_id'):
        # Get max FRP as indicator of fire intensity
        max_frp_idx = event_group['frp'].idxmax()
        max_frp_row = event_group.loc[max_frp_idx]
        
        # Create event summary
        event = {
            'event_id': event_id,
            'start_date': event_group['datetime'].min(),
            'end_date': event_group['datetime'].max(),
            'duration_days': (event_group['datetime'].max() - event_group['datetime'].min()).days + 1,
            'num_detections': len(event_group),
            'latitude': max_frp_row['latitude'],  # Use coordinates of max intensity
            'longitude': max_frp_row['longitude'],
            'max_frp': event_group['frp'].max(),
            'avg_frp': event_group['frp'].mean(),
            'total_frp': event_group['frp'].sum(),
            'max_confidence': event_group['confidence'].max(),
            'avg_confidence': event_group['confidence'].mean(),
            'year': max_frp_row['year'],
            'month': max_frp_row['datetime'].month,
            'day': max_frp_row['datetime'].day,
        }
        
        event_data.append(event)
    
    # Create new dataframe for events
    events_df = pd.DataFrame(event_data)
    
    # Convert start and end dates back to datetime
    events_df['start_date'] = pd.to_datetime(events_df['start_date'])
    events_df['end_date'] = pd.to_datetime(events_df['end_date'])
    
    print(f"Created events dataframe with {len(events_df)} fire events")
    return events_df

def save_processed_data(df, name, year):
    """Save processed data to CSV."""
    output_dir = f'../data/processed/{year}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{name}_{year}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {name} data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and preprocess wildfire and weather data')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2019)')
    parser.add_argument('--wildfire_file', type=str, default='../data/raw/ca_daily_fire_2000_2021-v2.csv', 
                        help='Path to consolidated California wildfires file')
    parser.add_argument('--save_intermediate', action='store_true', 
                        help='Save intermediate data before preprocessing')
    parser.add_argument('--cluster_distance', type=float, default=10.0,
                        help='Distance threshold in km for clustering wildfire events')
    parser.add_argument('--weather_radius', type=float, default=50.0,
                        help='Radius in km to search for weather stations around fire events')
    args = parser.parse_args()
    
    year = args.year
    print(f"Processing data for year {year}")
    
    # Step 1: Load and preprocess wildfire data
    print("Step 1: Loading and preprocessing wildfire data")
    wildfires_raw = load_wildfire_data(year, args.wildfire_file)
    processed_wildfires = preprocess_wildfire_data(wildfires_raw)
    
    # Step 2: Cluster wildfire data into events
    print("Step 2: Clustering wildfire data into events")
    wildfire_events = cluster_wildfire_events(processed_wildfires, args.cluster_distance)
    
    # Save intermediate wildfire data if requested
    if args.save_intermediate:
        if wildfires_raw is not None:
            save_processed_data(wildfires_raw, 'wildfires_raw', year)
        if processed_wildfires is not None:
            save_processed_data(processed_wildfires, 'wildfires', year)
        if wildfire_events is not None:
            save_processed_data(wildfire_events, 'wildfire_events_intermediate', year)
        print("Intermediate wildfire data saved")
    
    # Step 3: Load weather data near fire events
    print("Step 3: Loading weather data near fire events")
    weather_raw, stations = load_weather_data(year, fire_locations=wildfire_events, radius_km=args.weather_radius)
    
    # Step 4: Preprocess weather data
    print("Step 4: Preprocessing weather data")
    processed_weather = preprocess_weather_data(weather_raw)
    
    # Save final processed data
    if wildfire_events is not None:
        save_processed_data(wildfire_events, 'wildfire_events', year)
    if processed_weather is not None:
        save_processed_data(processed_weather, 'weather', year)
    if stations is not None:
        save_processed_data(stations, 'stations', year)
    
    print("Data processing complete!")