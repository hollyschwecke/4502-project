#aggregated_analysis.py
"""Aggregated visualization module for multi-year wildfire analysis."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def aggregate_visualizations():
    """Create visualizations that aggregate data across multiple years (2011-2020)."""
    
    # Setup output directory
    output_dir = '../results/aggregate'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and combine data from all years
    all_features = []
    all_importance = []
    all_metrics = []
    years = range(2011, 2021)  # 2011-2020
    
    for year in years:
        # Load feature data
        feature_path = f'../data/features/{year}/wildfire_event_features_{year}.csv'
        if os.path.exists(feature_path):
            df = pd.read_csv(feature_path)
            df['year'] = year
            all_features.append(df)
        
        # Load importance data
        importance_path = f'../models/{year}/feature_importance.csv'
        if os.path.exists(importance_path):
            df = pd.read_csv(importance_path)
            df['year'] = year
            all_importance.append(df)
            
        # Look for metrics file
        metrics_path = f'../models/{year}/xgboost_metadata.json'  # Updated path based on new save_model function
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            metrics_df = pd.DataFrame([metrics])
            all_metrics.append(metrics_df)
    
    if not all_features:
        print("No feature data found for any year")
        return
    
    # Combine data
    combined_features = pd.concat(all_features, ignore_index=True)
    print(f"Loaded data for {len(combined_features)} wildfires across {len(all_features)} years")
    
    # Determine intensity column based on what's available
    frp_col = next((col for col in ['fire_max_frp', 'fire_avg_frp', 'max_frp', 'fire_frp'] 
                   if col in combined_features.columns), None)
    
    if not frp_col:
        print("No fire intensity column found. Available columns:", combined_features.columns)
        return
    
    print(f"Using {frp_col} as fire intensity measure")
    
    # Determine month column
    month_col = next((col for col in ['fire_month', 'month'] if col in combined_features.columns), None)
    if not month_col and 'fire_start_date' in combined_features.columns:
        combined_features['month'] = pd.to_datetime(combined_features['fire_start_date']).dt.month
        month_col = 'month'
    
    # 1. Monthly Distribution Across All Years
    if month_col:
        plt.figure(figsize=(12, 6))
        monthly_counts = combined_features[month_col].value_counts().sort_index()
        plt.bar(monthly_counts.index, monthly_counts.values, color='darkred')
        plt.title('Wildfire Event Distribution by Month (2011-2020)')
        plt.xlabel('Month')
        plt.ylabel('Number of Wildfire Events')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monthly_distribution_all_years.png')
        plt.close()
    
    # 3. Feature Importance Across Years
    if all_importance:
        combined_importance = pd.concat(all_importance, ignore_index=True)
        
        # Get top 10 features based on average importance
        top_features = combined_importance.groupby('Feature')['Importance'].mean().nlargest(10).index
        
        # Filter to top features
        top_importance = combined_importance[combined_importance['Feature'].isin(top_features)]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_importance.groupby('Feature')['Importance'].mean().reset_index().sort_values('Importance', ascending=False))
        plt.title('Top 10 Features for Fire Intensity Prediction (2011-2020)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_all_years.png')
        plt.close()
        
        # 4. Feature importance trends over years
        plt.figure(figsize=(14, 8))
        top5_features = top_features[:5] 
        for feature in top5_features:
            feature_data = combined_importance[combined_importance['Feature'] == feature]
            feature_by_year = feature_data.groupby('year')['Importance'].mean()
            plt.plot(feature_by_year.index, feature_by_year.values, marker='o', linewidth=2, label=feature)
        
        plt.title('Feature Importance Trends Over Years')
        plt.xlabel('Year')
        plt.ylabel('Importance Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(sorted(combined_importance['year'].unique()))
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_trends.png')
        plt.close()
    
    # 5. Weather vs Fire Intensity - Aggregated Scatter Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for col, pos, title in [
        ('avg_temp_30d', (0, 0), 'Temperature'),
        ('total_precip_30d', (0, 1), 'Precipitation'),
        ('days_without_rain', (1, 0), 'Days Without Rain'),
        ('avg_dewpoint', (1, 1), 'Dewpoint')  # Changed to dewpoint since you mentioned its importance
    ]:
        if col in combined_features.columns:
            # Scatter plot with alpha for density visualization
            axes[pos].scatter(
                combined_features[col], 
                combined_features[frp_col], 
                alpha=0.2,
                s=10,
                c=combined_features['year'] if 'year' in combined_features.columns else 'red'
            )
            axes[pos].set_title(f'Fire Intensity vs {title}')
            axes[pos].set_xlabel(title)
            axes[pos].set_ylabel('Fire Intensity (FRP)')
            
            # Add trend line
            if len(combined_features) > 1:
                z = np.polyfit(combined_features[col], combined_features[frp_col], 1)
                p = np.poly1d(z)
                x_sorted = sorted(combined_features[col])
                axes[pos].plot(x_sorted, p(x_sorted), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weather_vs_fire_all_years.png')
    plt.close()
    
    # 9. Focus on dewpoint relationship
    if 'avg_dewpoint' in combined_features.columns and frp_col:
        plt.figure(figsize=(14, 8))
        
        # Create main scatter plot
        sns.scatterplot(x='max_wind_speed', y=frp_col, data=combined_features, 
                       alpha=0.3, s=40, hue=month_col if month_col else None)
        
    
        
        plt.title('Fire Intensity vs Average Windspeed (2011-2020)')
        plt.xlabel('Average Windspeed')
        plt.ylabel('Fire Intensity (FRP)')
        plt.grid(True, alpha=0.3)
        
        # Add annotation explaining relationship
        slope = z[0]
        correlation = combined_features[['avg_wind_speed', frp_col]].corr().iloc[0,1]
        plt.annotate(f"Correlation: {correlation:.2f}\nSlope: {slope:.2f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/windspeed_vs_intensity.png')
        plt.close()
    
    print(f"All aggregate visualizations saved to {output_dir}")

if __name__ == "__main__":
    print("Creating aggregated visualizations for years 2011-2021")
    aggregate_visualizations()