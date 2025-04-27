#visualization.py

"""Visualization module for wildfire prediction project."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def create_visualizations(year):
    """Create visualizations of wildfire data and model results."""
    
    # Setup directories and load data
    output_dir = f'../results/{year}/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        wildfire_features = pd.read_csv(f'../data/features/{year}/wildfire_event_features_{year}.csv')
        importance_df = pd.read_csv(f'../models/{year}/feature_importance.csv')
        print(f"Loaded data with {len(wildfire_features)} wildfire events")
        
        # Print column names to help with debugging
        print("Available columns:", wildfire_features.columns.tolist())
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        return
    
    # Determine intensity column based on what's available
    frp_col = next((col for col in ['fire_max_frp', 'fire_avg_frp', 'max_frp'] 
                   if col in wildfire_features.columns), None)
    
    if not frp_col:
        print("No fire intensity column found. Available columns:", wildfire_features.columns)
        return
    
    print(f"Using {frp_col} as fire intensity measure")
    
    # Determine month column
    month_col = next((col for col in ['fire_month', 'month'] if col in wildfire_features.columns), None)
    if not month_col and 'fire_start_date' in wildfire_features.columns:
        wildfire_features['month'] = pd.to_datetime(wildfire_features['fire_start_date']).dt.month
        month_col = 'month'
    
    # 1. Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title(f'Top Features - {year}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    # 2. Monthly Distribution
    if month_col:
        plt.figure(figsize=(10, 6))
        month_counts = wildfire_features[month_col].value_counts().sort_index()
        sns.barplot(x=month_counts.index, y=month_counts.values)
        plt.title(f'Wildfire Distribution by Month - {year}')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monthly_distribution.png')
        plt.close()
    
    # 3. Fire Intensity vs Top Feature
    if not importance_df.empty:
        top_feature = importance_df.iloc[0]['Feature']
        if top_feature in wildfire_features.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=top_feature, y=frp_col, data=wildfire_features, 
                           hue=month_col if month_col else None)
            plt.title(f'Fire Intensity vs {top_feature} - {year}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/frp_vs_top_feature.png')
            plt.close()
    
    # 4. Weather vs Fire Intensity
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    for col, pos, title in [
        ('avg_temp_30d', (0, 0), 'Temperature'),
        ('total_precip_30d', (0, 1), 'Precipitation'),
        ('days_without_rain', (1, 0), 'Days Without Rain'),
        ('max_wind_speed', (1, 1), 'Wind Speed')
    ]:
        if col in wildfire_features.columns:
            sns.scatterplot(x=col, y=frp_col, data=wildfire_features, ax=ax[pos])
            ax[pos].set_title(f'Fire Intensity vs {title}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weather_vs_fire.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()
    
    create_visualizations(args.year)