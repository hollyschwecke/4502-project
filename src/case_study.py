#!/usr/bin/env python3
"""
Simplified case study analysis for wildfire prediction project.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def analyze_wildfires(features_df, year, count=5):
    """Analyze the most significant wildfire events."""
    if features_df is None or features_df.empty:
        print("No feature data available for analysis")
        return
    
    # Create output directory
    output_dir = f'../results/{year}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Select most intense fires
    top_fires = features_df.sort_values('fire_frp', ascending=False).head(count)
    
    print(f"\nAnalyzing {count} most intense wildfires of {year}:")
    
    # Create a summary
    summary_rows = []
    
    # Analyze each fire
    for idx, fire in top_fires.iterrows():
        print(f"\nFire on {fire['fire_date'].split()[0]} at ({fire['fire_latitude']:.4f}, {fire['fire_longitude']:.4f})")
        print(f"Fire Radiative Power: {fire['fire_frp']:.2f}")
        print(f"Nearest Weather Station: {fire['station_name']} ({fire['station_distance_km']:.1f} km)")
        
        # Weather conditions
        print(f"Weather Conditions (30 days prior):")
        print(f"  - Avg Temperature: {fire['avg_temp_30d']:.1f}°F")
        print(f"  - Max Temperature: {fire['max_temp_30d']:.1f}°F")
        print(f"  - Total Precipitation: {fire['total_precip_30d']:.2f} inches")
        print(f"  - Days Without Rain: {fire['days_without_rain']}")
        print(f"  - Avg Wind Speed: {fire['avg_wind_speed']:.1f} mph")
        print(f"  - Max Wind Speed: {fire['max_wind_speed']:.1f} mph")
        
        # Add to summary
        summary_rows.append({
            'Date': fire['fire_date'].split()[0],
            'Latitude': fire['fire_latitude'],
            'Longitude': fire['fire_longitude'],
            'FRP': fire['fire_frp'],
            'Avg_Temp': fire['avg_temp_30d'],
            'Max_Temp': fire['max_temp_30d'],
            'Total_Rain': fire['total_precip_30d'],
            'Dry_Days': fire['days_without_rain'],
            'Max_Wind': fire['max_wind_speed']
        })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'top_wildfires.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot relationship between dry days and temperature for top fires
    plt.scatter(summary_df['Dry_Days'], summary_df['Max_Temp'], s=summary_df['FRP']/2, alpha=0.7)
    
    # Add labels for each fire
    for i, row in summary_df.iterrows():
        plt.annotate(row['Date'], (row['Dry_Days'], row['Max_Temp']))
    
    plt.title(f'Top Wildfires of {year}: Relationship Between Dry Days, Temperature, and Fire Intensity')
    plt.xlabel('Days Without Rain (30 days prior)')
    plt.ylabel('Maximum Temperature (°F)')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'top_wildfires.png'))
    plt.close()
    
    print(f"\nAnalysis saved to {output_dir}")
    return summary_df

def load_features(year):
    """Load feature data for analysis."""
    features_path = f'../data/features/{year}/wildfire_features_{year}.csv'
    try:
        features = pd.read_csv(features_path)
        print(f"Loaded feature data with {len(features)} records")
        return features
    except FileNotFoundError:
        print(f"Feature data not found at {features_path}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze wildfire case studies')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2019)')
    parser.add_argument('--count', type=int, default=5, help='Number of top wildfires to analyze')
    args = parser.parse_args()
    
    year = args.year
    count = args.count
    
    # Load feature data
    features = load_features(year)
    
    if features is not None:
        # Analyze wildfires
        analyze_wildfires(features, year, count)
    
    print("Case study analysis complete!")